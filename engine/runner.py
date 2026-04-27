def run_pipelines(context, pipeline_map):
    results = []

    for name in context.pipeline_candidates:
        if name not in pipeline_map:
            raise ValueError(f"Pipeline {name} not found")

        pipeline_fn = pipeline_map[name]

        try:
            result = pipeline_fn(context.input)
        except Exception as e:
            # graceful failure
            result = {
                "pipeline": name,
                "output": "",
                "latency": 999,   # heavy penalty
                "cost": 0,
                "error": str(e)
            }

        results.append(result)

    context.results = results
    return context