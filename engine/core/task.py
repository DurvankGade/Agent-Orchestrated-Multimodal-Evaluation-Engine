from typing import Callable, Optional

class Task:
    def __init__(self, 
                 name: str, 
                 instruction: str, 
                 metric_fn: Callable[[str, str], float],
                 postprocess_fn: Optional[Callable[[str], str]] = None):
        """
        name: task identifier
        instruction: prompt to the model
        metric_fn: evaluation function (pred, gt) -> score
        postprocess_fn: optional cleanup function for model output
        """
        self.name = name
        self.instruction = instruction
        self.metric_fn = metric_fn
        self.postprocess_fn = postprocess_fn

    def format_prompt(self, input_data: str) -> str:
        # Use simple string interpolation for the instruction
        # Instruction should contain {input} placeholder
        if "{input}" in self.instruction:
            return self.instruction.replace("{input}", input_data)
        return f"{self.instruction}\n\nInput: {input_data}"

    def __repr__(self):
        return f"Task(name={self.name})"
