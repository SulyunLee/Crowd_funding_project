# Crowd funding project
## Summary
This project uses web-scraped data from the Indiegogo crowdfunding platform (https://www.indiegogo.com/) to explore the association between entrepreneurs' responses to the delays of product delivery and backers' sentiment. We grouped the entrepreneurs' responses to delays into four categories: apology, ignore, promise, and transparency. We used these four delay response categories and related control variables to predict the backers' sentiment scores that are measured using the Python TextBlob sentiment analysis model. We used the linear regression models for the explanatory model analysis.

## Model description
We used linear regression models to predict the backers' sentiment scores for the delays of products using the following independent variables and control variables. We used the variance inflation factor (VIF) to rule out the possible collinearity and log-transformation for the highly skewed variables.

### Dependent variable
- The average sentiment scores estimated by the sentiment analysis of backers' comments after the delay.

### Independent variable
- Delay response - apology: founder apologizes for the delay
- Delay response - ignore: founder does not mention about the delay
- Delay response - promise: founder makes a promise or guarantee related to the delivery or production
- Delay response - transparency: founder provides a great deal of description about the founder's further actions

### Control variable
- Founder type - team: The project is owned by a team entrepreneurs
- Founder type - male: The project is owned by a male individual
- Initial funding: the initial amount of funding for the project
- Initial backers: the initial number of backers for the project
- Tech product: The project produces a high-technology product (e.g., wearables, health&fitness, and bluetooth)
