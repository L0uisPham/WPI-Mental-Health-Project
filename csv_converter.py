import pandas as pd

read_file = pd.read_excel ("new_data/WHS WPI Shared Data  20221004 Update.xlsx")
  
read_file.to_csv ("new_data/WPI_WH_Data_Pre.csv", 
                  index = None,
                  header=True)
    
df = pd.DataFrame(pd.read_csv("new_data/WPI_WH_Data_Pre.csv"))
  
df