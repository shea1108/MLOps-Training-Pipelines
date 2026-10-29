from zenml import step


@step
def deployment_trigger(r2_score: float, min_r2: float = 0.0) -> bool:
    return r2_score > min_r2
