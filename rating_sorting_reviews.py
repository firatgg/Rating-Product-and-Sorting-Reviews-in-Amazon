###################################################
# TASK 1: Calculate the Average Rating according to Current Reviews and Compare it with the Existing Average Rating.
###################################################

# In the shared dataset, users rated and commented on a product.
# Our aim in this task is to evaluate the scores by weighting them according to the date.
# The initial average score should be compared with the weighted score according to the date to be obtained.


###################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("C:/projects/pythonProject/data_sets/amazon_review.csv")
df = df_.copy()
df.head()
df.info()
df["overall"].mean()


###################################################
# Step 2: Calculate the Weighted Grade Point Average by Date.
###################################################
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

# Step 3:  Compare and interpret the average of each time period in weighted scoring

# averaging 28% of the first 30 days of evaluations

df.loc[df["day_diff"] <= 30, "overall"].mean() * 28 / 100

# average of the 2nd and 3rd month of evaluations with a weight of 26%

df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26 / 100

# evaluations from month 3 and up to month 6 with a weighted average of 24%

df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24 / 100

# 22% weighted average of evaluations after 6 months

df.loc[(df["day_diff"] > 180), "overall"].mean() * 22 / 100

# The averages of the evaluations are predominantly decreasing with the passage of time.

###################################################
# Task 2: Determine the 20 Reviews that will be displayed on the Product Detail Page for the Product.
####################################################

###################################################
# Step 1. Create the helpful_no Variable
###################################################

# Note:
# total_vote is the total number of up-down votes given to a comment.
# up means helpful.
# There is no helpful_no variable in the dataset, it needs to be generated from existing variables.
df.info()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df.head()
df["helpful_yes"].value_counts()
df[["helpful_yes", "total_vote", "helpful_no"]].head(50)

###################################################
# Step 2. Calculate and Add score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores to the Data
###################################################


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

    - The lower limit of the confidence interval to be calculated for the
       Bernoulli parameter p is considered as the WLB score.
    - The score is used for product ranking.
    - Note
    If the scores are between 1-5, 1-3 can be marked as negative and
    4-5 as positive and can be made Bernoulli compatible.
    This brings some problems with it. For this reason, it is necessary to make a Bayesian average rating.

Translated with www.DeepL.com/Translator (free version)

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head(50)

##################################################
# Step 3. Identify 20 Comments and Interpret the Results.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# The first 4 comments are old and useful, 5 and 6 are new but can be expected to be useful and
# can be expected to go higher and higher.
