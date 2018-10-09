def prediction(model, model_name, feature_weights):
    print('##### Model',model_name,'#####')

    # Cross Validation
    cv[model_name] = cross_val_score(model, # Cross-validation on model
                             X, # Feature matrix
                             y, # Output vector
                             cv=kf, # Cross-validation technique
                             scoring='accuracy' # Model performance metrics: accuracy
                            )
    print('Report Average Cross-Validation Accuracy of',model_name+':')
    print(np.mean(cv[model_name])*100, '%')

    # Fit the model on train data
    model.fit(X=X_train, y=y_train)

    # Predict outputs for test data
    y_pred = model.predict(X_test)

    # Create confusion matrix
    cm[model_name] = confusion_matrix(y_test, y_pred)
    print('\nConfusion Matrix of',model_name)
    print(cm[model_name])
    print("Confusion Matrix Prediction Accuracy: ", accuracy_score(y_test, y_pred)*100,'%')

    # ROC and AUC
    # Get predicted scores Pr(y=1): Used as thresholds for calculating TP Rate and FP Rate
    # model.classes_
    score = model.predict_proba(X_test)[:, 1]

    # Plot ROC Curve
    fpr[model_name], tpr[model_name], thresholds = roc_curve(y_test, score) # fpr: FP Rate, tpr: TP Rate, thresholds: Pr(y=1)
    roc_auc[model_name] = auc(fpr[model_name], tpr[model_name])

    plt.plot(fpr[model_name], tpr[model_name], label='AUC =coeff %0.2f'% roc_auc[model_name])
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.title('Receiver operating characteristic')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Get feature weights
    if feature_weights:
        if feature_weights == "coeff":
            weights = pd.DataFrame(model.coef_[0])
        elif feature_weights == "features":
            weights = pd.DataFrame(model.feature_importances_)
        # Bagging method makes use of many decision trees
        elif feature_weights == "bagging":
            weights = np.mean([tree.feature_importances_ for tree in model.estimators_], axis = 0)
            weights = pd.DataFrame(weights)

        coeffs[model_name] = pd.merge(pd.DataFrame(features_list),weights, left_index=True, right_index=True, how='left')
        coeffs[model_name].columns = ['Feature', 'Weights']
        coeffs[model_name] = coeffs[model_name].sort_values(by = 'Weights')

        plt.barh(coeffs[model_name]['Feature'], coeffs[model_name]['Weights'])
        plt.title('Feature Weights')
        plt.show()

        coeffs[model_name] = coeffs[model_name].sort_values(by = 'Weights', ascending=False)
        print(coeffs[model_name])
