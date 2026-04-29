def planner_node(state):

    if state["modality"] == "text":

        # dynamic decision
        if len(state["input"]) < 20:
            state["pipeline_candidates"] = ["simple", "phi3"]
        else:
            state["pipeline_candidates"] = ["mistral", "phi3"]

    elif state["modality"] == "image":
        state["pipeline_candidates"] = ["tesseract", "easyocr"]

    elif state["modality"] == "audio":
        state["pipeline_candidates"] = ["whisper"]
    
    print(f"[Planner] Selected: {state['pipeline_candidates']}")

    return state