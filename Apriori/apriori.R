# install.packages('arules');
library(arules);
data(Groceries);

frequentItems <- eclat (Groceries, parameter = list(supp = 0.1, maxlen = 15))
inspect(frequentItems)

rules_supp <- sort (rules, by="support", decreasing=TRUE)
inspect(head(rules_supp))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE)
inspect(head(rules_conf))

rules_lift <- sort (rules, by="lift", decreasing=TRUE)
inspect(head(rules_lift))

# Interesting rules
# {soda} => {other vegetables}
# {yogurt} => {root vegetables}
# {canned fish,hygiene articles} => {whole milk}
# {sausage} => {shopping bags}
# {shopping bags} => {soda}