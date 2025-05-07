# Prediction test using custom-made Linear Regression Model LR1 (Trained by me!)
# When we say pre-trained then it is trained by others and I am using that model

import LR1 as lr1

study_hours = float(input("Hours of study: "))
predicted_score = lr1.model.predict(study_hours)
print(f"Expected score: {predicted_score:.2f}")