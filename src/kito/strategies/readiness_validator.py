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

    @staticmethod
    def check_pretrained_weights_config(weight_path):
        """
        Validate pretrained weights configuration.

        Args:
            weight_path: path to pretrained weights

        Raises:
            ValueError: If weight_load_path not specified
            FileNotFoundError: If weight file doesn't exist
            ValueError: If weight file has wrong extension or is invalid
            PermissionError: If weight file is not readable
        """

        # Check path specified
        if not weight_path or weight_path == '':
            raise ValueError(
                "initialize_model_with_saved_weights=True but weight_load_path is not specified!\n"
                "Please provide config.model.weight_load_path = '/path/to/weights.pt'"
            )

        # Convert to Path for validation
        from pathlib import Path
        weight_file = Path(weight_path)

        # Check file exists
        if not weight_file.exists():
            raise FileNotFoundError(
                f"Pretrained weight file not found: '{weight_path}'\n"
                f"Please check the path in your configuration.\n"
                f"Expected file at: {weight_file.absolute()}"
            )

        # Check it's a file (not directory)
        if not weight_file.is_file():
            raise ValueError(
                f"Weight path is not a file: '{weight_path}'\n"
                f"Expected a .pt file, got a directory."
            )

        # Check extension
        if weight_file.suffix != '.pt':
            raise ValueError(
                f"Invalid weight file extension: '{weight_file.suffix}'\n"
                f"Expected .pt file, got: {weight_file.name}\n"
                f"Valid PyTorch weight files must have .pt extension."
            )

        # Check file is readable
        import os
        if not os.access(weight_file, os.R_OK):
            raise PermissionError(
                f"Weight file exists but is not readable: '{weight_path}'\n"
                f"Check file permissions."
            )
