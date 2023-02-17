from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from import_trait_data import TARGET_COLUMN, BINARY_VARS, TRAITS_WITH_NANS, CONTINUOUS_VARS

apriori_features_to_target_encode = ['Genus', 'Family']
all_features_to_target_encode = apriori_features_to_target_encode + ['kg_mode']


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


def knn_imputer(train_data: pd.DataFrame, test_data: pd.DataFrame, unlabelled: pd.DataFrame = None):
    from sklearn.impute import KNNImputer
    imp = KNNImputer(missing_values=np.nan)

    if unlabelled is not None:

        train_data_to_use = pd.concat([train_data, unlabelled])
        imp.fit(train_data_to_use)

    else:
        imp.fit(train_data)

    out = imp.transform(test_data)
    out_df = pd.DataFrame(out, columns=test_data.columns, index=test_data.index)

    return out_df


def do_basic_preprocessing(X: pd.DataFrame, y: pd.DataFrame, train_index=None, test_index=None,
                           unlabelled_data: pd.DataFrame = None,
                           impute: bool = True, variance: float = None,
                           categorical_features: List[str] = None,
                           scale: bool = True, PCA_cont_vars: bool = False):
    from sklearn.feature_selection import VarianceThreshold

    import category_encoders as ce
    from sklearn.compose import ColumnTransformer
    # use copies
    X_copy = X.copy(deep=True)
    y_copy = y.copy(deep=True)
    if train_index is not None and test_index is not None:
        # Note iloc select by position rather than index label
        X_train, X_test = X_copy.iloc[train_index], X_copy.iloc[test_index]
        y_train, y_test = y_copy.iloc[train_index], y_copy.iloc[test_index]
    else:
        X_train = X_copy
        X_test = X_copy
        y_train = y_copy
        y_test = y_copy

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
        bin_features_to_encode = [c for c in X_copy.columns if c in BINARY_VARS]
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


    if scale:
        cols_to_scale = [x for x in variance_X_train.columns if x != TARGET_COLUMN]
        # Scale data using unlabelled data.
        standard_scaler = ColumnTransformer(
            transformers=[
                ("standard_scaling", StandardScaler(),
                 cols_to_scale)
            ],
            remainder='passthrough', verbose_feature_names_out=False

        )
        all_selected_data = pd.concat([variance_X_train, variance_unlabelled])
        standard_scaler.fit(all_selected_data)
        X_train_scaled = pd.DataFrame(standard_scaler.transform(variance_X_train),
                                      index=variance_X_train.index,
                                      columns=standard_scaler.get_feature_names_out())
        X_test_scaled = pd.DataFrame(standard_scaler.transform(variance_X_test),
                                     index=variance_X_test.index,
                                     columns=standard_scaler.get_feature_names_out())

        if unlabelled_data is not None:
            unlabelled_scaled = pd.DataFrame(standard_scaler.transform(variance_unlabelled),
                                             index=variance_unlabelled.index,
                                             columns=standard_scaler.get_feature_names_out())
        else:
            unlabelled_scaled = variance_unlabelled
    else:
        X_train_scaled = variance_X_train
        X_test_scaled = variance_X_test
        unlabelled_scaled = variance_unlabelled

    # Impute missing data values
    if impute:
        if not scale:
            raise ValueError('Better to scale before imputing')

        imputed_X_train = knn_imputer(X_train_scaled, X_train_scaled, unlabelled=unlabelled_scaled)
        imputed_X_test = knn_imputer(X_train_scaled, X_test_scaled, unlabelled=unlabelled_scaled)

        if unlabelled_data is not None:
            imputed_unlabelled = knn_imputer(X_train_scaled, unlabelled_scaled,
                                             unlabelled=unlabelled_scaled)
        else:
            imputed_unlabelled = unlabelled_scaled
    else:

        imputed_X_train = X_train_scaled
        imputed_X_test = X_test_scaled
        imputed_unlabelled = unlabelled_scaled

    if PCA_cont_vars:
        if not scale:
            raise ValueError('Better to scale before PCA')
        cols_to_PCA = [x for x in CONTINUOUS_VARS if x in imputed_X_train.columns]
        # Scale data using unlabelled data.
        standard_PCA = ColumnTransformer(
            transformers=[
                ("PCA_cont_vars", PCA(n_components=0.8),
                 cols_to_PCA)
            ],
            remainder='passthrough', verbose_feature_names_out=False

        )
        all_selected_data = pd.concat([imputed_X_train, imputed_unlabelled])
        standard_PCA.fit(all_selected_data)
        X_train_pcad = pd.DataFrame(standard_PCA.transform(imputed_X_train),
                                    index=imputed_X_train.index,
                                    columns=standard_PCA.get_feature_names_out())
        X_test_pcad = pd.DataFrame(standard_PCA.transform(imputed_X_test),
                                   index=imputed_X_test.index,
                                   columns=standard_PCA.get_feature_names_out())

        if unlabelled_data is not None:
            unlabelled_pcad = pd.DataFrame(standard_PCA.transform(imputed_unlabelled),
                                           index=imputed_unlabelled.index,
                                           columns=standard_PCA.get_feature_names_out())
        else:
            unlabelled_pcad = imputed_unlabelled
        if unlabelled_data is not None:
            pd.testing.assert_index_equal(unlabelled_data.index, unlabelled_pcad.index)
        pd.testing.assert_index_equal(X_train.index, X_train_pcad.index)
        pd.testing.assert_index_equal(X_test.index, X_test_pcad.index)
        return X_train_pcad, X_test_pcad, unlabelled_pcad
    else:
        pd.testing.assert_index_equal(X_train.index, imputed_X_train.index)
        pd.testing.assert_index_equal(X_test.index, imputed_X_test.index)
        if unlabelled_data is not None:
            pd.testing.assert_index_equal(unlabelled_data.index, imputed_unlabelled.index)
        return imputed_X_train, imputed_X_test, imputed_unlabelled
