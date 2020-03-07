# install.packages('arules');
library(arules);
data(Groceries);

frequentItems <- eclat (Groceries, parameter = list(supp = 0.07, maxlen = 15)) # calculates support for frequent items
inspect(frequentItems)

rules <- apriori (Groceries, parameter = list(supp = 0.001, conf = 0.5))

rules_conf <- sort (rules, by="confidence", decreasing=TRUE) # 'high-confidence' rules.
inspect(head(rules_conf))

rules_lift <- sort (rules, by="lift", decreasing=TRUE) # 'high-lift' rules.
inspect(head(rules_lift))

rules <- apriori (data=Groceries, parameter=list (supp=0.001,conf = 0.15,minlen=2), appearance = list(default="rhs",lhs="soda"), control = list (verbose=F)) # those who bought 'milk' also bought..
rules_conf <- sort (rules, by="count", decreasing=TRUE) # 'high-confidence' rules.
inspect(head(rules_conf))
