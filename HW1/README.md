# Dataset Documentation
## Data source
I used web crawlers to get the passenger traffic of High-Speed Rail Station data from [THSR official website](https://www.thsrc.com.tw/corp/9571df11-8524-4935-8a46-0d5a72e6bc7c). The website provides the number of arrival and departure passengers every month in the 12 stations from 2017-01 to 2025-01. I get the data from 2022-01 to 2025-01 to train and test the model.<br><br>
In addition, specific holidays will affect the passenger traffic, so I also include it in the training data. (hardcode in the program when data preprocess). Specific holidays (疏運期間) and the adjusted ticketing service can also be found on [THSR website](https://www.thsrc.com.tw/ArticleContent/60dbfb79-ac20-4280-8ffb-b09e7c94f043).<br><br>
As for number of people diagnosed with COVID-19, it’s from [government open data website](https://data.gov.tw/dataset/160838) and then calculate to get the correct time range.

## Data description
The dataset consists of 37 months, 12 stations, and a total of 444 rows. `thsr_raw0.csv`, which includes only [Date, Station id, Crowd number], delivers better performance. In contrast, `thsr_raw1.csv`, which contains [Date, Station id, Crowd number, Schedule number, Disease number], results in poorer predictions.

| Column name | Data type | Description | Example |
| ----------- | --------- | ----------- | ------- |
| Date | String | Month from 2022-01 to 2025-01. Format: yyyy-mm | 	2025-01|
| Station id	| String | Chinese name of the 12 stations | 南港|
| Crowd number | Integer | Passenger traffic of that month | 360796|
|Schedule number | Integer | Number of train schedule every week | 1103|
| Disease number | Integer | Number of people diagnosed with COVID-19 | 0|

After preprocessing, the final data looks like the following.<br>

| Column name | Data type | Description | Example |
| ----------- | --------- | ----------- | ------- |
| Date |Integer | Month from 2022-01 to 2025-01. | 202312|
| Station id | Float | Average crowd flow in that station | 247475.9|
| Crowd number | Integer | Passenger traffic of that month | 332100|
| Schedule number | Integer | Number of train schedule every week | 1060|
| Disease number | Integer | Number of people diagnosed with COVID-19 | 14|
| Special days | Integer | Number of specific holidays in that month, ex: new year, Double Tenth | 3 |

## Data collection and preprocess
### Web crawlers
I use web crawlers to collect the data and save it in `thsr_raw.csv`. The related program is in `data.py`. Run this file to get the raw data.
### Factors that may affect the crowd number
- **Holiday**: <br>People tend to go home or go out to play in holidays. THSR also have some policies regarding holidays. There are always much more people in the station when it’s holidays, so the number of holidays in a month is definitely one of the factors that will affect the crowd number.
- **Diseases (ex: COVID-19)**: <br>If there’s some serious disease recently, people tend not to take public transportation. In 2021.5, when lots of people were diagnosed with COVID-19, the number of people taking THSR has a significantly decrease.
- **Number of available trains**: <br>The more train available, the more passenger it can take. THSR has increased *the number of train schedule every week* five times since 2022 to meet the need of so many passengers.
### Preprocess
When training the model, the program will first call `Data_preprocess` function in `utils.py` to get the processed data (store as data frame and return to main function).<br>
- **Add number of holidays in the month to the data:**<br>
As I mentioned above, holiday is an important factor that will affect the passenger traffic, so I add a new column in the dataset to record the days of the holiday.
- **Data splitting:**
I split the dataset into training data, validation data, and testing data. Each contain few months and all stations’ crowd number. The splitting is continuous, not random, because time is meaningful in crowd flow prediction. We can use the past few months’ data to predict the next month, but not using future data to determine the historical data. If we split the data randomly, the time will become meaningless.<br>

    | Data | Range | Amount |
    | ---- | ----- | ------ |
    |Training | 2022-01 ~ 2023-12 | 24 months (288 rows, 65% of the data)|
    |Validation | 2024-01 ~ 2024-06 | 6 months (72 rows, 16% of the data)|
    |Testing | 2024-07 ~ 2025-01 | 7 months (84 rows, 19% of the data)|

- **Station id:**
At first, I use 1~12 to represent the stations, but later I found that using the average crowd number in that station can represent the stations better. Some stations such as Taipei have more passengers than others. Using average crowd number in that station as station id can provide more information to the model.