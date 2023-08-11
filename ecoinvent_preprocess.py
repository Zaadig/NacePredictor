import pandas as pd

file_path = 'src/Database-Overview-for-ecoinvent-v3.9.1.xlsx'
df_activities = pd.read_excel(file_path, sheet_name='Consequential AO')
df_intermediate = pd.read_excel(file_path, sheet_name='Intermediate Exchanges')


merged_df = pd.merge(df_activities, df_intermediate, on='CPC Classification', how='left')

desired_columns = [
    'Activity Name', 'Special Activity Type', 'Sector', 'ISIC Classification',
    'Name', 'By-product Classification', 'Product Information_x'
]
result_df = merged_df[desired_columns]


result_df.to_excel('dst/ecoinvent_preprocessed.xlsx', index=False)
