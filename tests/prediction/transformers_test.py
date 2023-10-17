import unittest
import numpy as np

from insurance_charges_model.prediction.transformers import BooleanTransformer


class TransformersTests(unittest.TestCase):

    def test_boolean_transformer(self):
        # arrange
        boolean_transformer = BooleanTransformer(true_value="yes", false_value="no")
        X = [['yes'], ['no'], ['yes']]

        # act
        boolean_transformer.fit(X)
        result = boolean_transformer.transform(X)

        # assert
        self.assertTrue((result == np.array([[True], [False], [True]])).all())

    def test_boolean_transformer_with_boolean_values(self):
        # arrange
        boolean_transformer = BooleanTransformer(true_value="yes", false_value="no")
        X = [[True], [False], [False]]

        # act
        boolean_transformer.fit(X)
        result = boolean_transformer.transform(X)

        # assert
        self.assertTrue((result == np.array([[True], [False], [False]])).all())

    def test_boolean_transformer_with_bad_values(self):
        # arrange
        boolean_transformer = BooleanTransformer(true_value="yes", false_value="no")
        X = [['yes'], ['no'], ['yes']]

        # act, assert
        boolean_transformer.fit(X)

        with self.assertRaises(ValueError):
            result = boolean_transformer.transform([['asd'], ['no'], ['yes']])


if __name__ == '__main__':
    unittest.main()