class Context:
    def __init__(self, input_data, modality):
        self.input = input_data
        self.modality = modality  # "text" or "image"

        self.pipeline_candidates = []
        self.results = []

        self.best = None