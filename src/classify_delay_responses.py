
import pandas as pd

if __name__ == "__main__":
    original_project_filename = "temp_data/rf_predictions_ntree128_md32_delay_response_success.csv"
    original_project_df = pd.read_csv(original_project_filename)
    print(original_project_df.shape)

    # remove rows with response NA values
    original_project_df.dropna(subset=["response_apology", "response_promise", "response_ignore", "response_transparency"], inplace=True)
    print(original_project_df.shape)
