import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#df=pd.read_csv('data/immoweb-dataset.csv')

class Immo_Preprocessing:

    def __init__(self, file_path):
            self.df = pd.read_csv(file_path)

    def cleaning_columns(self):
        # Clean column names: remove whitespace, lower case, replace space by underscore
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Drop unused columns
        self.df.drop(columns=['unnamed:_0', 'url', 'id', 'monthlycost', 'hasbalcony', 'accessibledisabledpeople',
                        'bathroomcount','roomcount','hasattic','hasbasement','hasdressingroom',
                        'diningroomsurface','hasdiningroom','floorcount','streetfacadewidgth',
                        'floodzonetype','heatingtype','hasheatpump','hasvoltaicpanels',
                        'hasthermicpanels','kitchensurface','kitchentype','haslivingroom',
                        'livingroomsurface','gardenorientation','hasairconditioning',
                        'hasarmoreddoor','hasvisiophone','hasoffice',
                        'hasfireplace','terracesurface','terraceorientation',
                        'gardensurface', 'toiletcount',
                        'hasphotovoltaicpanels', 'streetfacadewidth','buildingconstructionyear',
                        'facedecount', 'landsurface'], inplace=True, errors='ignore') 

    def fill_boolean_values(self):
        # Fill boolean values with False
        self.df[['hasswimmingpool', 'haslift', 'hasgarden', 'hasterrace']] = self.df[['hasswimmingpool', 'haslift', 'hasgarden', 'hasterrace']].fillna(False)
        # Converts types
        self.df[['haslift','hasgarden','hasswimmingpool', 'hasterrace']] = self.df[['haslift','hasgarden','hasswimmingpool', 'hasterrace']].astype(int)

    # Add parking column
    def add_parking_col(self):
            def __check_if_too_large(row, col):
                return row[col] > 1000

            def __check_if_large_and_apt(row, col):
                return row[col] > 100 and row['type'] == 'APARTMENT'

            orig = self.df.shape[0]
            rows_to_drop = self.df.apply(lambda row: __check_if_too_large(row, 'parkingcountindoor') or __check_if_too_large(row, 'parkingcountoutdoor'), axis=1)
            self.df = self.df[~rows_to_drop]
            print(f'Dropped {orig - self.df.shape[0]} too large')

            orig = self.df.shape[0]
            rows_to_drop = self.df.apply(lambda row: __check_if_large_and_apt(row, 'parkingcountindoor') or __check_if_large_and_apt(row, 'parkingcountoutdoor'), axis=1)
            self.df = self.df[~rows_to_drop]
            print(f'Dropped {orig - self.df.shape[0]} large and apartment type')

            self.df["hasparking"] = self.df.apply(
                lambda row: 1 if (
                    (not pd.isna(row.parkingcountindoor) and row.parkingcountindoor > 0)
                    or (not pd.isna(row.parkingcountoutdoor) and row.parkingcountoutdoor > 0)
                ) else 0,
                axis=1
            )
            print(self.df['hasparking'].value_counts())


    # Add Region column
    def add_region_column(self):
        province_to_region = {
            'Antwerp': 'Flanders', 'East Flanders': 'Flanders',
            'Flemish Brabant': 'Flanders', 'Limburg': 'Flanders',
            'West Flanders': 'Flanders', 'Walloon Brabant': 'Wallonia',
            'Hainaut': 'Wallonia', 'Liège': 'Wallonia',
            'Luxembourg': 'Wallonia', 'Namur': 'Wallonia',
            'Brussels': 'Brussels'
        }
        self.df['region'] = self.df['province'].map(province_to_region)


    # eliminate outliers
    def eliminate_outliers(self):
        cols = ['bedroomcount', 'habitablesurface', 'price']

        for col in cols:
            # 75th percentile
            seventy_fifth = self.df[col].quantile(0.75)
            # 25th percentile
            twenty_fifth = self.df[col].quantile(0.25)
            # Interquartile range
            surface_iqr = seventy_fifth - twenty_fifth

            # Upper threshold
            upper = seventy_fifth + (1.5 * surface_iqr)
            # Lower threshold
            lower = twenty_fifth - (1.5 * surface_iqr)

            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]

        self.df=self.df[~self.df.index.isin(outliers.index)]

        self.df = self.df[~self.df['habitablesurface'].isna()]
        self.df['habitablesurface'] = self.df['habitablesurface'].astype(float)
        self.df = self.df.drop(self.df[self.df['habitablesurface'] > 1500].index)
        self.df = self.df.drop(self.df[self.df['bedroomcount'] > 20].index)   

    def eliminate_postcode_small_number(self):
        # Eliminates postcodes < 10 properties
        pc_stats = self.df.groupby('postcode')['postcode'].agg('count').sort_values(ascending=False)
        pc_stats_less_than10 = pc_stats[pc_stats <= 10]
        self.df.postcode = self.df.postcode.apply(lambda x: 'other' if x in pc_stats_less_than10 else x)
        print(len(self.df.postcode.unique()))
        self.df = self.df[self.df.postcode != 'other']

    def drop_remaining_rows(self):
        # Drop rows 
        self.df.dropna(subset=["price"], inplace=True)
        self.df.dropna(subset=["habitablesurface"], inplace=True)
        self.df.dropna(subset=["epcscore"], inplace=True)
        self.df = self.df[self.df['epcscore'].str.contains("_") == False]
        self.df = self.df[self.df['epcscore'].str.contains("X") == False]
        self.df.dropna(subset=["buildingcondition", "bedroomcount"], inplace=True)

    def oe_ohe_processing(self):
        # Pre processing EPC Score and Building condition
        oe = OrdinalEncoder(categories=[['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']])
        self.df['epcscore_encoded'] = oe.fit_transform(self.df[['epcscore']])

        oe2 = OrdinalEncoder(categories=[['AS_NEW','JUST_RENOVATED','GOOD','TO_BE_DONE_UP','TO_RENOVATE','TO_RESTORE']])
        self.df['buildingcondition_encoded'] = oe2.fit_transform(self.df[['buildingcondition']])

        # One-Hot Encoding Region
        ohe = OneHotEncoder(sparse_output=False)
        encoded = ohe.fit_transform(self.df[['region']])

        df_one_hot = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['region']), index=self.df.index)
        self.df = pd.concat([self.df, df_one_hot], axis=1)

        # Preprocessing Type
        self.df['type_encoded'] = (self.df['type'] == 'HOUSE').astype(int)
    
    def drop_remaining_columns(self):
        # Drop columns 
        self.df.drop(columns=['parkingcountindoor','parkingcountoutdoor'], inplace=True)
        self.df.drop(columns='subtype', inplace=True)
        self.df.drop(columns=['type','province','locality','buildingcondition','epcscore','region'], inplace=True)
        
        self.df.dropna(inplace=True)
    
    def drop_duplicates(self):
    # Drop duplicates
        self.df.drop_duplicates(inplace=True)
        
    def latitude_longitude_columns(self, filepath2):
        df_geo = pd.read_csv(filepath2, sep=';')
        #'data/georef-belgium-postal-codes@public.csv'
        pc_geo_dict = {}
        for pc in self.df['postcode'].unique():
            geo_point = df_geo.loc[df_geo['Post code'] == pc, 'Geo Point'].values
            if len(geo_point) > 0:
                pc_geo_dict[pc] = geo_point[0]
            else:
                pc_geo_dict[pc] = None
        self.df['geocode'] = self.df['postcode'].map(pc_geo_dict)

        self.df[['latitude', 'longitude']] = self.df['geocode'].str.split(',', expand=True)

        self.df['latitude'] = self.df['latitude'].astype(float)
        self.df['longitude'] = self.df['longitude'].astype(float)

        self.df.drop(columns=['geocode'], inplace=True)
        self.df.drop(columns=['postcode'], inplace=True)
        
    def save(self, output_path):
            self.df.to_csv(output_path, index=False)
            #df.to_csv('data_cleaned.csv')
            
    def data_pre_processing(self):
        self.cleaning_columns()
        self.fill_boolean_values()
        self.add_parking_col()
        self.add_region_column()
        self.eliminate_outliers()
        self.eliminate_postcode_small_number()
        self.drop_remaining_rows()
        self.oe_ohe_processing()
        self.drop_remaining_columns()
        self.drop_duplicates()
        self.latitude_longitude_columns('data/georef-belgium-postal-codes@public.csv')
        self.df.info()

if __name__ == "__main__":
    processor = Immo_Preprocessing('data/immoweb-dataset.csv')
    processor.data_pre_processing()
    processor.save('data/data_cleaned.csv')
