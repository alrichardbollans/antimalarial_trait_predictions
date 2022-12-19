from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from import_trait_data import TARGET_COLUMN, BINARY_VARS, DISCRETE_VARS, TRAITS_WITH_NANS


def basic_data_prep(train_data: pd.DataFrame, traits_to_use: List[str], dropna_cols=False, dropna_rows=False,
                    ):
    if dropna_cols:
        train_data = train_data.dropna(axis=1)

    trait_cols = [x for x in traits_to_use if x in train_data.columns]
    print(f'Using traits: {trait_cols}')
    train_data = train_data[trait_cols + [TARGET_COLUMN]]
    if dropna_rows:
        train_data = train_data.dropna(subset=trait_cols, axis=0, how='any')

    X = train_data[trait_cols]
    y = train_data[TARGET_COLUMN]

    return X, y


def get_semi_supervised_data(X_train: pd.DataFrame, y_train: pd.DataFrame,
                             unlabelled_data_to_use: pd.DataFrame):
    """
    Sometimes specific unlabelled data should be supplied in order to match encodings fo varaibles etc.
    :param X_train:
    :param y_train:
    :param unlabelled_data_to_use:
    :return:
    """

    unlabelled_data = unlabelled_data_to_use

    all_data = pd.concat([X_train, unlabelled_data])

    if y_train is not None:

        unlabelled_data[TARGET_COLUMN] = -1
        all_y = pd.concat([y_train, unlabelled_data[TARGET_COLUMN]])
    else:
        all_y = None

    return all_data, all_y


def knn_imputer(train_data: pd.DataFrame, test_data: pd.DataFrame, unlabelled: pd.DataFrame = None,
                vars_to_rediscretise=None):
    from sklearn.impute import KNNImputer
    imp = KNNImputer(missing_values=np.nan)

    if unlabelled is not None:

        train_data_to_use = pd.concat([train_data, unlabelled])
        imp.fit(train_data_to_use)

    else:
        imp.fit(train_data)

    out = imp.transform(test_data)
    out_df = pd.DataFrame(out, columns=test_data.columns, index=test_data.index)

    # Set discrete vars to ints (note some variables that are usually discrete may be target encoded etc..)
    if vars_to_rediscretise is not None:
        for c in vars_to_rediscretise:
            if c in out_df.columns:
                out_df[c] = out_df[c].astype(int)

    return out_df


def do_basic_preprocessing(X: pd.DataFrame, y: pd.DataFrame, train_index=None, test_index=None,
                           unlabelled_data: pd.DataFrame = None,
                           impute: bool = True, variance: float = None,
                           categorical_features: List[str] = None,
                           scale: bool = True):
    from sklearn.feature_selection import VarianceThreshold

    import category_encoders as ce
    from sklearn.compose import ColumnTransformer

    if train_index is not None and test_index is not None:
        # Note iloc select by position rather than index label
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y


    if (train_index is not None and test_index is None) or (test_index is not None and train_index is None):
        raise ValueError


    # Target encode categorical features
    # Defaults to mean when transforming unknown values
    if categorical_features is not None:
        for c in categorical_features:
            if c not in TRAITS_WITH_NANS:
                # If possible to convert to float, raise error as these should be strings
                try:
                    test1 = X_train[c].astype(float)
                except ValueError:
                    pass
                else:
                    raise ValueError(f'Trying to target encode floats: {c}. Have they already been encoded?')
        target_encoder = ce.TargetEncoder(cols=categorical_features)
        target_encoder.fit(X_train, y_train)
        encoded_X_train = target_encoder.transform(X_train)
        encoded_X_test = target_encoder.transform(X_test)
        if unlabelled_data is not None:
            encoded_unlabelled = target_encoder.transform(
                unlabelled_data)
        else:
            encoded_unlabelled = unlabelled_data
    else:

        encoded_X_train = X_train
        encoded_X_test = X_test
        encoded_unlabelled = unlabelled_data

    if variance is not None:
        # Remove binary features with 0 or 1 in variance% of the samples
        bin_features_to_encode = [c for c in X.columns if c in BINARY_VARS]
        bin_variance_selector = VarianceThreshold(threshold=(variance * (1 - variance)))
        variance_selection_transformer = ColumnTransformer(
            transformers=[
                ("bin_feature_selection_step", bin_variance_selector, bin_features_to_encode)],
            remainder='passthrough', verbose_feature_names_out=False

        )

        variance_selection_transformer.fit(encoded_X_train, y_train)
        cols_with_good_variance = variance_selection_transformer.get_feature_names_out()

        cols_with_bad_variance = [x for x in bin_features_to_encode if x not in cols_with_good_variance]
        print(
            f'Features removed due to low variance: {cols_with_bad_variance}')

        variance_X_train = pd.DataFrame(variance_selection_transformer.transform(encoded_X_train),
                                        index=encoded_X_train.index,
                                        columns=cols_with_good_variance)
        variance_X_test = pd.DataFrame(variance_selection_transformer.transform(encoded_X_test),
                                       index=encoded_X_test.index,
                                       columns=cols_with_good_variance)
        if unlabelled_data is not None:
            variance_unlabelled = pd.DataFrame(variance_selection_transformer.transform(encoded_unlabelled),
                                               index=encoded_unlabelled.index,
                                               columns=cols_with_good_variance)
        else:
            variance_unlabelled = encoded_unlabelled
    else:
        variance_X_train = encoded_X_train
        variance_X_test = encoded_X_test
        variance_unlabelled = encoded_unlabelled

    # Impute missing data values
    if impute:
        vars_to_rediscretise = [x for x in DISCRETE_VARS if x not in categorical_features]
        imputed_X_train = knn_imputer(variance_X_train, variance_X_train, unlabelled=variance_unlabelled,
                                      vars_to_rediscretise=vars_to_rediscretise)
        imputed_X_test = knn_imputer(variance_X_train, variance_X_test, unlabelled=variance_unlabelled,
                                     vars_to_rediscretise=vars_to_rediscretise)

        if unlabelled_data is not None:
            imputed_unlabelled = knn_imputer(variance_X_train, variance_unlabelled,
                                             unlabelled=variance_unlabelled,
                                             vars_to_rediscretise=vars_to_rediscretise)
        else:
            imputed_unlabelled = variance_unlabelled

    else:
        imputed_X_train = variance_X_train
        imputed_X_test = variance_X_test
        imputed_unlabelled = variance_unlabelled

    if scale:
        cols_to_scale = [x for x in imputed_X_train.columns if x != TARGET_COLUMN]
        # Scale data using unlabelled data.
        standard_scaler = ColumnTransformer(
            transformers=[
                ("standard_scaling", StandardScaler(),
                 cols_to_scale)
            ],
            remainder='passthrough', verbose_feature_names_out=False

        )
        all_selected_data = pd.concat([imputed_X_train, imputed_unlabelled])
        standard_scaler.fit(all_selected_data)
        X_train_scaled = pd.DataFrame(standard_scaler.transform(imputed_X_train),
                                      index=imputed_X_train.index,
                                      columns=standard_scaler.get_feature_names_out())
        X_test_scaled = pd.DataFrame(standard_scaler.transform(imputed_X_test),
                                     index=imputed_X_test.index,
                                     columns=standard_scaler.get_feature_names_out())

        if unlabelled_data is not None:
            unlabelled_scaled = pd.DataFrame(standard_scaler.transform(imputed_unlabelled),
                                             index=imputed_unlabelled.index,
                                             columns=standard_scaler.get_feature_names_out())
        else:
            unlabelled_scaled = imputed_unlabelled

        pd.testing.assert_index_equal(X_train.index, X_train_scaled.index)
        pd.testing.assert_index_equal(X_test.index, X_test_scaled.index)
        if unlabelled_data is not None:
            pd.testing.assert_index_equal(unlabelled_data.index, unlabelled_scaled.index)
        return X_train_scaled, X_test_scaled, unlabelled_scaled
    else:
        pd.testing.assert_index_equal(X_train.index, imputed_X_train.index)
        pd.testing.assert_index_equal(X_test.index, imputed_X_test.index)
        if unlabelled_data is not None:
            pd.testing.assert_index_equal(unlabelled_data.index, imputed_unlabelled.index)
        return imputed_X_train, imputed_X_test, imputed_unlabelled
