# Copyright 2025 ADogDev/BaconAI

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import requests
from typing import Dict
from datetime import datetime, timedelta
from .tool import Tool

class CryptoInfoTool(Tool):
    def __init__(self, api_key, TIME_FORMAT="%Y-%m-%d"):
        description = "Get historical cryptocurrency data"
        name = "crypto_info"
        parameters = {
            "crypto_symbol": {
                "type": "string",
                "description": "The symbol of the cryptocurrency (e.g., BTC, ETH)",
            },
            "currency": {
                "type": "string",
                "description": "The currency to compare against (e.g., USD, EUR)",
            },
            "start_date": {
                "type": "string",
                "description": "The start date for the historical data",
            },
            "end_date": {
                "type": "string",
                "description": "The end date for the historical data",
            },
        }
        super(CryptoInfoTool, self).__init__(description, name, parameters)
        self.TIME_FORMAT = TIME_FORMAT
        self.api_key = api_key

    def _parse(self, data):
        parsed_data: dict = {}
        for item in data["prices"]:
            date = datetime.utcfromtimestamp(item[0] / 1000).strftime(self.TIME_FORMAT)
            parsed_data[date] = {
                "price": item[1]
            }
        return parsed_data

    def _query(self, crypto_symbol, currency, start_date, end_date):
        """Query the cryptocurrency historical data API"""
        url = (
            f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart/range"
            f"?vs_currency={currency}&from={int(datetime.strptime(start_date, self.TIME_FORMAT).timestamp())}"
            f"&to={int(datetime.strptime(end_date, self.TIME_FORMAT).timestamp())}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        data = response.json()
        return self._parse(data)

    def func(self, crypto_dict: Dict) -> Dict:
        crypto_symbol = crypto_dict["crypto_symbol"].lower()
        currency = crypto_dict["currency"].lower()
        start_date = datetime.strftime(
            datetime.strptime(crypto_dict["start_date"], self.TIME_FORMAT),
            self.TIME_FORMAT,
        )
        end_date = crypto_dict.get("end_date")
        if end_date is None:
            end_date = datetime.strftime(
                datetime.strptime(start_date, self.TIME_FORMAT) + timedelta(days=1),
                self.TIME_FORMAT,
            )
        else:
            end_date = datetime.strftime(
                datetime.strptime(end_date, self.TIME_FORMAT),
                self.TIME_FORMAT,
            )
        if datetime.strptime(start_date, self.TIME_FORMAT) > datetime.strptime(end_date, self.TIME_FORMAT):
            start_date, end_date = end_date, start_date
        assert start_date != end_date, "Start date and end date must not be the same."
        return self._query(crypto_symbol, currency, start_date, end_date)
