**Performance of different Machine learning methods on predicting
the result of a football match**

Predictions for football matches have always been a notoriously difficult task because of the three way
outcome of a match, also called 1X2 betting. What is especially difficult is to determine the cases where
neither team has the edge, therefore predicting a draw. In this paper, different machine learning methods
are applied to tackle this task and to see whether some are better suited for this problem than others.
The dataset used for predictions consists of historical records, specifically the last 12 seasons of the English
Premier League (2014-2026). The dataset was enriched with additional features that were derived from the
existing ones, such as rolling averages of the features, points per game, team elo ranking and expected goals
(xG). The models were evaluated on their log loss and brier score. The models used could be divided into
discriminative and generative. The discriminative models included a feed-forward neural network, ensemble
methods as Random forest and Gradient boosting, multinomial Logistic regression and Support vector
machine. The generative models included Poisson regression from Dixon-Coles paper [7] and a bivariate
Poisson regression that is an extension of the former one first introduced by [12]. The main purpose of this
paper is to maximize the predictive performance of the generative Poisson model and compare it to other
common methods in the industry, such as the ensembles, as a reference point. All the models were evaluated
on their performance of the 1X2 prediction task and compared to the bookmakers’ odds as well as a baseline
model that predicts the most likely outcome based on the historical distribution of the target variable. The
results show that all the models fall short of the bookmakers’ odds, however, the best performing model
was the bivariate Poisson regression with covariates, which was just marginally underpreforming the market.
This was the result of finding ways via algorithmic improvements as well as reducing the features space.
Despite the other models trailing by a larger margin, they all very comfortably outperform the baseline
model, which is a clear indication that the methods used in this paper are able to capture the signal in the
data and learn patterns that are relevant for the prediction task. The results of this paper can be used as a
reference point for future research in this area, as well as a benchmark for other models that are used in the
industry.
