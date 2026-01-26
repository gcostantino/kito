class ReadinessValidator:
    """
    Validates module readiness for different operations.

    This replaces the decorator pattern with a cleaner strategy pattern
    that is easier to test and extend.

    Usage:
        # In Engine
        ReadinessValidator.check_for_training(module)
        ReadinessValidator.check_for_inference(module, weight_path)
    """

    @staticmethod
    def check_for_training(module):
        """
        Check if module is ready for training.

        Args:
            module: BaseModule instance

        Raises:
            RuntimeError: If module is not ready
        """
        if not module.is_built:
            raise RuntimeError(
                f"Module '{module.model_name}' not built. "
                "Call module.build() before training."
            )

        if not module.is_optimizer_set:
            raise RuntimeError(
                f"Module '{module.model_name}' optimizer not set. "
                "Call module.associate_optimizer() before training."
            )

        if module.learning_rate is None:
            raise RuntimeError(
                f"Module '{module.model_name}' learning_rate not set."
            )

    @staticmethod
    def check_for_inference(module, weight_path=None):
        """
        Check if module is ready for inference.

        Args:
            module: BaseModule instance
            weight_path: Optional weight path to check

        Raises:
            RuntimeError: If module is not ready
        """
        if not module.is_built:
            raise RuntimeError(
                f"Module '{module.model_name}' not built. "
                "Call module.build() before inference."
            )

        if not module.is_weights_loaded:
            raise RuntimeError(
                f"Module '{module.model_name}' weights not loaded. "
                "Call module.load_weights() or engine.load_weights() before inference."
            )

        if weight_path is not None:
            import os
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Weight file not found: {weight_path}")

    @staticmethod
    def check_data_loaders(train_loader=None, val_loader=None, test_loader=None):
        """
        Check if data loaders are provided.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader

        Raises:
            ValueError: If required loaders are missing
        """
        if train_loader is None and val_loader is None and test_loader is None:
            raise ValueError("At least one data loader must be provided")
