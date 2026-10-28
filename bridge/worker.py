"""One Linux worker per durable volume. RabbitMQ inbox handoff then GPU processing."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import aio_pika
from config import load
from contracts import validate, event
from job_store import JobStore
from processor import process

async def main():
    config=load()
    state=Path(os.environ.get('STATE_DIR','state'));state.mkdir(parents=True,exist_ok=True)
    lock=(state/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    store=JobStore(state/'jobs.sqlite3')
    connection=await aio_pika.connect_robust(os.environ['AMQP_URL'],heartbeat=60)
    async with connection:
        consumer=await connection.channel();await consumer.set_qos(prefetch_count=1)
        publisher=await connection.channel(publisher_confirms=True,on_return_raises=True)
        exchange=await publisher.get_exchange('identity-exchange',ensure=True)
        # Topology is owned by BE. Fail at startup if expected queues do not exist.
        await publisher.get_queue('ai.analysis.result',ensure=True)
        async def receive(message):
            try:
                request=json.loads(message.body);validate(request);store.accept(request)
            except (ValueError,KeyError,TypeError):
                print(json.dumps({'stage':'reject_invalid_or_conflicting_request'}),flush=True)
                await message.reject(requeue=False);return
            except Exception:
                await message.nack(requeue=True);return
            await message.ack()  # durable SQLite owns the job now
        for kind in ('image','video'):
            queue=await consumer.get_queue(f'ai.analysis.server.{kind}.requested',ensure=True)
            await queue.consume(receive)
        while True:
            row=store.pending()
            if row is None:
                await asyncio.sleep(1);continue
            identifier,body,attempts,saved=row; request=json.loads(body)
            if saved is None:
                if attempts>=3:
                    result=event(request,error='WORKER_ATTEMPTS_EXHAUSTED')
                else:
                    store.attempt(identifier)
                    try: result=await asyncio.to_thread(process,request,config)
                    except Exception as exc:
                        print(json.dumps({'request_id':identifier,'stage':'processing_failed','error':type(exc).__name__}),flush=True)
                        await asyncio.sleep(5);continue
                store.save_result(identifier,result)
            else: result=json.loads(saved)
            try:
                await exchange.publish(aio_pika.Message(json.dumps(result).encode(),
                    content_type='application/json',delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    message_id=result['EventId'],correlation_id=identifier),
                    routing_key='identity.event.aianalysisresultevent',mandatory=True,timeout=30)
                store.published(identifier)
                print(json.dumps({'request_id':identifier,'stage':'result_confirmed'}),flush=True)
            except Exception as exc:
                print(json.dumps({'request_id':identifier,'stage':'publish_retry','error':type(exc).__name__}),flush=True)
                await asyncio.sleep(5)

if __name__=='__main__': asyncio.run(main())
