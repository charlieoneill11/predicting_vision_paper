base_path = "~/Documents/Github/paper"
data_path = "/input"

TRAINING_FILE = [base_path + data_path + "/df_1_years.csv", 
                 base_path + data_path + "/df_2_years.csv",
                 base_path + data_path + "/df_3_years.csv"]
EVALUATION_FILE = [base_path + data_path + "/test_1_years.csv",
                   base_path + data_path + "/test_2_years.csv",
                   base_path + data_path + "/test_3_years.csv"]
MULTIVISITS_FILE = [base_path + data_path + "/df_3_years_2_visits.csv",
                    base_path + data_path + "/df_3_years_3_visits.csv",
                    base_path + data_path + "/df_3_years_4_visits.csv",
                    base_path + data_path + "/df_3_years_5_visits.csv",
                    base_path + data_path + "/df_3_years_6_visits.csv",
                    base_path + data_path + "/df_3_years_7_visits.csv",
                    base_path + data_path + "/df_3_years_8_visits.csv"]
MODEL_OUTPUT = base_path+"/models/"