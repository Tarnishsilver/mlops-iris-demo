from pycaret.classification import (
    setup, compare_models, finalize_model, save_model
)
from pycaret.datasets import get_data

# Load the Iris dataset
iris = get_data("iris")

# Initialize the classification experiment
setup(
    data=iris,
    target="species",
    session_id=42,
    verbose=False,
    html=False
)

# Compare algorithms and select the best
best_model = compare_models()
final_model = finalize_model(best_model)

# Save the trained model
save_model(final_model, "iris_model")

print("Model trained and saved successfully!")
