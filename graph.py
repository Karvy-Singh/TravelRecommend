import matplotlib.pyplot as plt

# Methods and corresponding NDCG@7 values
methods = ['xgboost', 'logisticregression', 'random_forest_classifier', 'random_forest_regressor']
ndcg = [1.0, 0.9848, 0.9881, 0.9986]

plt.figure()
plt.bar(methods, ndcg)
plt.ylim(0.98, 1.0)  
plt.xlabel('Method')
plt.ylabel('NDCG@7')
plt.title('Comparison of NDCG@7 Across Methods')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

