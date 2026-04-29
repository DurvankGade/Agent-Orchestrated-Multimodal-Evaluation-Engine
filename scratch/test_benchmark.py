from pipelines.text.benchmark_api import run as benchmark_api

res = benchmark_api("test", "test instruction")
print(res)
