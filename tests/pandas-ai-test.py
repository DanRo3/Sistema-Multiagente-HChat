import os
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import pandas as pd


class ConversationalData:

    def __init__(self, data):
        self.df = data
        # self.preprocess_data_dropna_mostfreq('c_phone_prefix')
        # self.preprocess_data_dropna_mostfreq('c_phone')
        # self.preprocess_data_dropna_mostfreq('zone_name')
        # self.preprocess_data_dropna_mostfreq('status')
        # self.correct_date()
        # self.fix_na_mail()
        self.open_api_key = os.environ.get('OPENAI_API_KEY', '')
        self.llm = OpenAI(api_token=self.open_api_key, model_name="gpt-4o-mini")
        self.dfs = SmartDataframe(self.df, config={'llm': self.llm, "verbose": True, "conversational": True})

    def get_data(self):
        return self.df

    def get_model(self):
        return self.dfs

    def set_data(self, new_data):
        self.df = new_data

    def preprocess_data_dropna_mostfreq(self, column):
        value = self.df[column].value_counts()
        most = list(value.to_dict().keys())[0]
        self.df[column] = self.df[column].fillna(most)

    def make_username(self, full_name):
        splited = full_name.lower().rsplit(maxsplit=1)
        first_names, last_name = full_name.lower().rsplit(maxsplit=1) if len(splited) > 1 else [splited[0], ""]
        return first_names[0] + last_name + "@gmail.com" if len(splited) > 1 else first_names + "@gmail.com"

    def fix_na_mail(self):
        self.df["c_fname"] = self.df["c_fname"].fillna('')
        self.df["c_lname"] = self.df["c_lname"].fillna('')
        self.df['full_name'] = self.df.apply(lambda x: '%s %s' % (x['c_fname'], x['c_lname']), axis=1)
        for index, row in self.df.iterrows():
            if pd.isnull(self.df.at[index, 'c_email']):
                self.df.at[index, 'c_email'] = self.make_username(row['full_name'])

    def correct_date(self):
        self.df['dt'] = pd.to_datetime(self.df['dt'], format='%d/%m/%Y %H:%M')
        self.df['dt_to'] = pd.to_datetime(self.df['dt_to'], format='%d/%m/%Y %H:%M')
        self.df['modified'] = pd.to_datetime(self.df['modified'], format='%d/%m/%Y %H:%M')


path = os.environ.get('DATA_PATH','data/DataLimpia.csv')
data = pd.read_csv(path, sep=';', encoding='latin1')
cds = ConversationalData(data)

query=input(">:")

promt=f"Responda en español la siguiente consulta: {query}"
dfs = cds.get_model()
result = dfs.chat(promt)
print(result)