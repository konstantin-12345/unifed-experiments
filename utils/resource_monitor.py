import psutil
import time

class ResourceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_cpu = None
        self.start_memory = None

    def start_monitoring(self):
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent(interval=0.1)
        self.start_memory = psutil.virtual_memory().used

    def get_measurements(self):
        duration = time.time() - self.start_time
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_used = psutil.virtual_memory().used - self.start_memory
        # 简化的能耗估计
        energy = duration * cpu_usage / 100.0
        return {
            "duration": duration,
            "cpu_usage": cpu_usage,
            "memory_used": memory_used,
            "energy": energy
        }
