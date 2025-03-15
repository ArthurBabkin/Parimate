import threading
import queue
import time
class MultithreadHandler:
    @staticmethod
    def run_in_threads(functions, on_fi_done: callable):
        done_queue = queue.Queue()

        def wrapper(func, idx):
            output = func()
            done_queue.put((idx, output))

        threads = []
        for i, func in enumerate(functions, start=1):
            t = threading.Thread(target=wrapper, args=(func, i))
            t.start()
            threads.append(t)

        finished_count = 0
        while finished_count < len(functions):
            idx, output = done_queue.get()
            print(f"функция {idx} завершена!")
            on_fi_done(idx, output)
            finished_count += 1
