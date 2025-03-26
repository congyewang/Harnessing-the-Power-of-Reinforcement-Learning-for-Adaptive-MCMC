from typing import Any, Dict, List

import numpy as np
from cytoolz import pipe
from numpy import typing as npt
from posteriordb import PosteriorDatabase
from scipy.optimize import minimize

from .target import PosteriorDatabaseTargetPDF


class PosteriorDBToolbox:
    def __init__(self, posteriordb_path: str):
        self.posteriordb_path = posteriordb_path
        self.pdb = self.make_posteriordb()

    @staticmethod
    def flat(nested_list: List[List[Any]]) -> List[Any]:
        """
        Expand nested list

        Args:
            nested_list (List[List[Any]]): Nested list.

        Returns:
            List[Any]: Flattened list.
        """
        result: List[Any] = []
        stack = list(nested_list)

        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(reversed(item))
            else:
                result.append(item)

        return result[::-1]

    @staticmethod
    def merge_dict_list(
        list_dict_list: List[Dict[str, List[float]]],
    ) -> Dict[str, List[float]]:
        """
        Merge a list of dictionaries into a single dictionary.

        Args:
            list_dict_list (List[Dict[str, List[float]]]): List of dictionaries to merge.

        Returns:
            Dict[str, List[float]]: Merged dictionary.
        """
        if not list_dict_list:
            return {}

        result: Dict[str, List[float]] = {}

        for dict_list in list_dict_list:
            for key, value in dict_list.items():
                if key not in result:
                    result[key] = []
                result[key].extend(value)

        return result

    @staticmethod
    def convert_dict_to_ordered_array(
        merged_dict: Dict[str, List[float]],
    ) -> npt.NDArray[np.float64]:
        keys = merged_dict.keys()

        values_list = [merged_dict[key] for key in keys]

        return np.array(values_list).T

    def make_posteriordb(self) -> PosteriorDatabase:
        """
        Create a PosteriorDatabase object from the given path.
        """
        pdb = PosteriorDatabase(self.posteriordb_path)

        return pdb

    def get_model_name_with_gold_standard(self) -> List[str]:
        models = self.pdb.posterior_names()

        model_name_with_gold_standard: List[str] = []

        while models:
            try:
                model_name = models.pop()
                self.pdb.posterior(model_name).reference_draws()
                model_name_with_gold_standard.append(model_name)
            except AssertionError:
                pass

        return model_name_with_gold_standard

    def get_gold_standard(
        self,
        model_name: str,
    ) -> npt.NDArray[np.float64]:
        """
        Generate the gold standard for the given model.

        Args:
            model_name (str): Model name.
            posteriordb_path (str, optional): Path to the database. Defaults to os.path.join("posteriordb", "posterior_database").

        Returns:
            npt.NDArray[np.float64]: Gold standard.
        """
        ## Load Dataset
        posterior = self.pdb.posterior(model_name)

        ## Gold Standard
        gs_constrain = pipe(
            posterior.reference_draws(),
            self.merge_dict_list,
            self.convert_dict_to_ordered_array,
        )

        # Model Generation
        posterior_target = PosteriorDatabaseTargetPDF(model_name, self.posteriordb_path)
        model = posterior_target.make_model()

        gs_unconstrain = np.array(
            [model.param_unconstrain(np.array(i)) for i in gs_constrain]
        )

        return gs_unconstrain

    def get_fisher_information_matrix(self, model_name: str) -> npt.NDArray[np.float64]:
        """
        Get the negative inverse Hessian for the given model.

        Args:
            model_name (str): Model name.

        Returns:
            npt.NDArray[np.float64]: Negative inverse Hessian.
        """
        posterior_target = PosteriorDatabaseTargetPDF(model_name, self.posteriordb_path)
        log_target_pdf, grad_log_target_pdf, hess_log_target_pdf = (
            posterior_target.combine_make_log_target_pdf(mode=["pdf", "grad", "hess"])
        )

        model_dim = self.get_gold_standard(model_name).shape[1]
        init_point = np.random.normal(size=model_dim)

        maximum = minimize(
            lambda x: -log_target_pdf(x),
            init_point,
            jac=lambda x: -grad_log_target_pdf(x),
            hess=lambda x: -hess_log_target_pdf(x),
            method="trust-constr",
        )

        fisher_infermation_matrix = -np.linalg.inv(hess_log_target_pdf(maximum.x))

        return fisher_infermation_matrix
