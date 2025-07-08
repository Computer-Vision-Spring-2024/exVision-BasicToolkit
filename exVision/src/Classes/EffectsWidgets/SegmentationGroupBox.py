from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class SegmentationGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.segmentation_effect = None

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        ### ================= Region Growing ================= ###

        # Region Growing Threshold
        self.rg_threshold_HBoxLayout = QHBoxLayout()
        self.rg_threshold_HBoxLayout.setObjectName("")

        self.rg_threshold = QLabel()
        self.rg_threshold.setObjectName("region_growing_threshold")
        self.rg_threshold.setText("Threshold")

        self.rg_threshold_spinbox = QDoubleSpinBox()
        self.rg_threshold_spinbox.setObjectName("region_growing_threshold_slider")
        self.rg_threshold_spinbox.setValue(20)
        self.rg_threshold_spinbox.setSingleStep(1)
        self.rg_threshold_spinbox.setMinimum(1)
        self.rg_threshold_spinbox.setMaximum(100)

        self.rg_threshold_HBoxLayout.addWidget(self.rg_threshold)
        self.rg_threshold_HBoxLayout.addWidget(self.rg_threshold_spinbox)

        # Region Growing Window Size
        self.rg_window_size_HBoxLayout = QHBoxLayout()
        self.rg_window_size_HBoxLayout.setObjectName("window_size_HBoxLayout")

        self.rg_window_size_label = QLabel()
        self.rg_window_size_label.setObjectName("window_size_label")
        self.rg_window_size_label.setText("Window Size")

        self.rg_window_size_spinbox = QSpinBox()
        self.rg_window_size_spinbox.setObjectName("window_size_spinbox")
        self.rg_window_size_spinbox.setValue(3)
        self.rg_window_size_spinbox.setSingleStep(2)
        self.rg_window_size_spinbox.setMinimum(3)
        self.rg_window_size_spinbox.setMaximum(21)

        self.rg_window_size_HBoxLayout.addWidget(self.rg_window_size_label)
        self.rg_window_size_HBoxLayout.addWidget(self.rg_window_size_spinbox)

        # Reset Region Growing
        self.reset_region_growing = QPushButton()
        self.reset_region_growing.setObjectName("Reset")
        self.reset_region_growing.setText("Reset Region Growing")

        # Apply Region Growing
        self.apply_region_growing = QPushButton()
        self.apply_region_growing.setObjectName("apply_region_growing")
        self.apply_region_growing.setText("Apply Region Growing")

        ### ================= Agglomerative Clustering ================= ###

        # Number of Clusters (K)
        self.num_of_clusters_HBoxLayout = QHBoxLayout()
        self.num_of_clusters_HBoxLayout.setObjectName("horizontalLayout_35")

        self.num_of_clusters_label = QLabel()
        self.num_of_clusters_label.setObjectName("agglo_num_of_clusters_label")
        self.num_of_clusters_label.setText("Number of Clusters (K)")

        self.num_of_clusters_spinbox = QSpinBox()
        self.num_of_clusters_spinbox.setMinimum(2)
        self.num_of_clusters_spinbox.setMaximum(10)
        self.num_of_clusters_spinbox.setObjectName("num_of_clusters_spinBox")

        self.num_of_clusters_HBoxLayout.addWidget(self.num_of_clusters_label)
        self.num_of_clusters_HBoxLayout.addWidget(self.num_of_clusters_spinbox)

        # Initial Number of Clusters
        self.initial_num_of_clusters_HBoxLayout = QHBoxLayout()
        self.initial_num_of_clusters_HBoxLayout.setObjectName(
            "initial_num_of_clusters_HBoxLayout"
        )
        self.initial_num_of_clusters_label = QLabel()
        self.initial_num_of_clusters_label.setObjectName(
            "initial_num_of_clusters_label"
        )
        self.initial_num_of_clusters_label.setText("Initial num of clusters")

        self.initial_num_of_clusters_spinBox = QSpinBox()
        self.initial_num_of_clusters_spinBox.setMinimum(3)
        self.initial_num_of_clusters_spinBox.setObjectName(
            "initial_num_of_clusters_spinBox"
        )

        self.initial_num_of_clusters_HBoxLayout.addWidget(
            self.initial_num_of_clusters_label
        )
        self.initial_num_of_clusters_HBoxLayout.addWidget(
            self.initial_num_of_clusters_spinBox
        )

        # Distance Calculation Method
        self.distance_calculation_method_VBoxLayout = QVBoxLayout()
        self.distance_calculation_method_VBoxLayout.setObjectName(
            "distance_calculation_method"
        )

        self.distance_calculation_method_label = QLabel()
        self.distance_calculation_method_label.setObjectName(
            "distance_calculation_method_label"
        )
        self.distance_calculation_method_label.setText("Distance Calculation Method")

        self.distance_calculation_method_combobox = QComboBox()
        self.distance_calculation_method_combobox.setObjectName(
            "distance_calculation_method_combobox"
        )
        self.distance_calculation_method_combobox.addItem("distance between centroids")
        self.distance_calculation_method_combobox.addItem("max distance between pixels")

        self.distance_calculation_method_VBoxLayout.addWidget(
            self.distance_calculation_method_label
        )
        self.distance_calculation_method_VBoxLayout.addWidget(
            self.distance_calculation_method_combobox
        )

        # Downsampling Checkbox
        self.downsampling_checkbox = QCheckBox()
        self.downsampling_checkbox.setObjectName("downsampling")
        self.downsampling_checkbox.setText("Downsample image")

        # Downsampling Scale Factor
        self.scale_factor_HBoxLayout = QHBoxLayout()
        self.scale_factor_HBoxLayout.setObjectName("scale_factor_HBoxLayout")

        self.downsampling_scale_factor_label = QLabel()
        self.downsampling_scale_factor_label.setObjectName(
            "downsampling_scale_factor_label"
        )
        self.downsampling_scale_factor_label.setText("Scale Factor")
        self.downsampling_scale_factor_spinbox = QSpinBox()
        self.downsampling_scale_factor_spinbox.setObjectName(
            "downsampling_scale_factor"
        )
        self.downsampling_scale_factor_spinbox.setValue(4)
        self.downsampling_scale_factor_spinbox.setSingleStep(1)
        self.downsampling_scale_factor_spinbox.setMinimum(2)
        self.downsampling_scale_factor_spinbox.setMaximum(10)
        self.downsampling_scale_factor_spinbox.setEnabled(False)

        self.scale_factor_HBoxLayout.addWidget(self.downsampling_scale_factor_label)
        self.scale_factor_HBoxLayout.addWidget(self.downsampling_scale_factor_spinbox)

        # Apply Segmentation Button
        self.apply_segmentation = QPushButton()
        self.apply_segmentation.setObjectName("apply_segmentation")
        self.apply_segmentation.setText("Apply Segmentation")

        # Elapsed Time
        self.elapsed_time_agglomerative = QLabel()
        self.elapsed_time_agglomerative.setObjectName("elapsed_time_agglomerative")
        self.elapsed_time_agglomerative.setText("Elapsed Time is ")

        ### ================= K-Means ================= ###

        # n Clusers
        self.n_clusters_HBoxLayout = QHBoxLayout()
        self.n_clusters_HBoxLayout.setObjectName("n_clusters_HBoxLayout")

        self.n_clusters_label = QLabel()
        self.n_clusters_label.setObjectName("n_clusters_label")
        self.n_clusters_label.setText("n clusters")

        self.n_clusters_spinBox = QSpinBox()
        self.n_clusters_spinBox.setObjectName("n_clusters_spinBox")
        self.n_clusters_spinBox.setValue(4)
        self.n_clusters_spinBox.setSingleStep(1)
        self.n_clusters_spinBox.setMinimum(2)
        self.n_clusters_spinBox.setMaximum(30)

        self.n_clusters_HBoxLayout.addWidget(self.n_clusters_label)
        self.n_clusters_HBoxLayout.addWidget(self.n_clusters_spinBox)

        # Max Iterations
        self.k_means_max_iterations_HBoxLayout = QHBoxLayout()
        self.k_means_max_iterations_HBoxLayout.setObjectName(
            "k_means_max_iterations_HBoxLayout"
        )

        self.k_means_max_iterations_label = QLabel()
        self.k_means_max_iterations_label.setObjectName("k_means_max_iterations_label")
        self.k_means_max_iterations_label.setText("Max Iterations")

        self.k_means_max_iteratation_spinBox = QSpinBox()
        self.k_means_max_iteratation_spinBox.setObjectName(
            "k_means_max_iteratation_spinBox"
        )
        self.k_means_max_iteratation_spinBox.setValue(4)
        self.k_means_max_iteratation_spinBox.setSingleStep(1)
        self.k_means_max_iteratation_spinBox.setMinimum(2)
        self.k_means_max_iteratation_spinBox.setMaximum(30)

        self.k_means_max_iterations_HBoxLayout.addWidget(
            self.k_means_max_iterations_label
        )
        self.k_means_max_iterations_HBoxLayout.addWidget(
            self.k_means_max_iteratation_spinBox
        )

        # Spatial Segmentation
        self.spatial_segmentation = QCheckBox()
        self.spatial_segmentation.setObjectName("spatial_segmentation")
        self.spatial_segmentation.setText("Spatial Segmentation")

        # Weight
        self.spatial_segmentation_weight_HBoxLayout = QHBoxLayout()
        self.spatial_segmentation_weight_HBoxLayout.setObjectName(
            "spatial_segmentation_weight_HBoxLayout"
        )

        self.spatial_segmentation_weight_label = QLabel()
        self.spatial_segmentation_weight_label.setObjectName(
            "spatial_segmentation_weight_label"
        )
        self.spatial_segmentation_weight_label.setText("Spatial Segmentation Weight")

        self.spatial_segmentation_weight_spinbox = QDoubleSpinBox()
        self.spatial_segmentation_weight_spinbox.setObjectName(
            "spatial_segmentation_weight_spinbox"
        )
        self.spatial_segmentation_weight_spinbox.setValue(1)
        self.spatial_segmentation_weight_spinbox.setSingleStep(0.1)
        self.spatial_segmentation_weight_spinbox.setMinimum(0)
        self.spatial_segmentation_weight_spinbox.setMaximum(2)
        self.spatial_segmentation_weight_spinbox.setEnabled(False)

        self.spatial_segmentation_weight_HBoxLayout.addWidget(
            self.spatial_segmentation_weight_label
        )
        self.spatial_segmentation_weight_HBoxLayout.addWidget(
            self.spatial_segmentation_weight_spinbox
        )

        # Centroid Optimization
        self.centroid_optimization = QCheckBox()
        self.centroid_optimization.setObjectName("Centroid_Optimization")
        self.centroid_optimization.setText("Centroid Optimization")
        self.centroid_optimization.setChecked(True)

        # LUV_Conversion
        self.k_means_LUV_conversion = QCheckBox()
        self.k_means_LUV_conversion.setObjectName("LUV_conversion")
        self.k_means_LUV_conversion.setText("Convert to LUV")

        # Apply
        self.apply_k_means = QPushButton()
        self.apply_k_means.setObjectName("apply_k_means")
        self.apply_k_means.setText("Apply K-Means")

        ### ================= Mean Shift ================= ###

        # Window Size
        self.mean_shift_window_size_HBoxLayout = QHBoxLayout()
        self.mean_shift_window_size_HBoxLayout.setObjectName(
            "mean_shift_window_size_HBoxLayout"
        )

        self.mean_shift_window_size_label = QLabel()
        self.mean_shift_window_size_label.setObjectName("mean_shift_window_size_label")
        self.mean_shift_window_size_label.setText("Window Size")

        self.mean_shift_window_size_spinbox = QSpinBox()
        self.mean_shift_window_size_spinbox.setObjectName(
            "mean_shift_window_size_spinbox"
        )
        self.mean_shift_window_size_spinbox.setValue(200)
        self.mean_shift_window_size_spinbox.setSingleStep(10)
        self.mean_shift_window_size_spinbox.setMinimum(20)
        self.mean_shift_window_size_spinbox.setMaximum(1000)

        self.mean_shift_window_size_HBoxLayout.addWidget(
            self.mean_shift_window_size_label
        )
        self.mean_shift_window_size_HBoxLayout.addWidget(
            self.mean_shift_window_size_spinbox
        )

        # Sigma
        self.mean_shift_sigma_HBoxLayout = QHBoxLayout()
        self.mean_shift_sigma_HBoxLayout.setObjectName("mean_shift_sigma_HBoxLayout")

        self.mean_shift_sigma_label = QLabel()
        self.mean_shift_sigma_label.setObjectName("mean_shift_sigma_label")
        self.mean_shift_sigma_label.setText("Sigma")

        self.mean_shift_sigma_spinbox = QSpinBox()
        self.mean_shift_sigma_spinbox.setObjectName("mean_shift_sigma_spinbox")
        self.mean_shift_sigma_spinbox.setValue(20)
        self.mean_shift_sigma_spinbox.setSingleStep(5)
        self.mean_shift_sigma_spinbox.setMinimum(5)
        self.mean_shift_sigma_spinbox.setMaximum(100)

        self.mean_shift_sigma_HBoxLayout.addWidget(self.mean_shift_sigma_label)
        self.mean_shift_sigma_HBoxLayout.addWidget(self.mean_shift_sigma_spinbox)

        # Convergence Threshold
        self.mean_shift_threshold_HBoxLayout = QHBoxLayout()
        self.mean_shift_threshold_HBoxLayout.setObjectName(
            "mean_shift_sigma_HBoxLayout"
        )

        self.mean_shift_threshold_label = QLabel()
        self.mean_shift_threshold_label.setObjectName("mean_shift_threshold_label")
        self.mean_shift_threshold_label.setText("Convergence Threshold")

        self.mean_shift_threshold_spinbox = QSpinBox()
        self.mean_shift_threshold_spinbox.setObjectName("mean_shift_threshold_spinbox")
        self.mean_shift_threshold_spinbox.setValue(10)
        self.mean_shift_threshold_spinbox.setSingleStep(2)
        self.mean_shift_threshold_spinbox.setMinimum(1)
        self.mean_shift_threshold_spinbox.setMaximum(50)

        self.mean_shift_threshold_HBoxLayout.addWidget(self.mean_shift_threshold_label)
        self.mean_shift_threshold_HBoxLayout.addWidget(
            self.mean_shift_threshold_spinbox
        )

        # LUV_Conversion
        self.mean_shift_LUV_conversion = QCheckBox()
        self.mean_shift_LUV_conversion.setObjectName("LUV_conversion")
        self.mean_shift_LUV_conversion.setText("Convert to LUV")

        # Apply Mean Shift
        self.apply_mean_shift = QPushButton()
        self.apply_mean_shift.setObjectName("apply_mean_shift")
        self.apply_mean_shift.setText("Apply Mean Shift")

        # Create a GroupBox and add Region Growing things inside it
        self.rg_groupbox = QGroupBox()
        self.rg_groupbox.setTitle("Region Growing")
        self.rg_groupbox.setObjectName("region_growing")
        self.rg_groupbox_VBoxLayout = QVBoxLayout()
        self.rg_groupbox_VBoxLayout.setObjectName("rg_groupbox_VBoxLayout")
        self.rg_groupbox.setLayout(self.rg_groupbox_VBoxLayout)
        self.rg_groupbox_VBoxLayout.addLayout(self.rg_threshold_HBoxLayout)
        self.rg_groupbox_VBoxLayout.addLayout(self.rg_window_size_HBoxLayout)
        self.rg_groupbox_VBoxLayout.addWidget(self.apply_region_growing)
        self.rg_groupbox_VBoxLayout.addWidget(self.reset_region_growing)

        # Create a GroupBox and add Agglomerative Clustering things inside it
        self.agglo_groupbox = QGroupBox()
        self.agglo_groupbox.setTitle("Agglomerative Clustering")
        self.agglo_groupbox.setObjectName("agglomerative_clustering")
        self.agglo_groupbox_VBoxLayout = QVBoxLayout()
        self.agglo_groupbox_VBoxLayout.setObjectName("agglo_groupbox_VBoxLayout")
        self.agglo_groupbox.setLayout(self.agglo_groupbox_VBoxLayout)
        self.agglo_groupbox_VBoxLayout.addLayout(self.num_of_clusters_HBoxLayout)
        self.agglo_groupbox_VBoxLayout.addLayout(
            self.initial_num_of_clusters_HBoxLayout
        )
        self.agglo_groupbox_VBoxLayout.addLayout(
            self.distance_calculation_method_VBoxLayout
        )
        self.agglo_groupbox_VBoxLayout.addWidget(self.downsampling_checkbox)
        self.agglo_groupbox_VBoxLayout.addLayout(self.scale_factor_HBoxLayout)
        self.agglo_groupbox_VBoxLayout.addWidget(self.apply_segmentation)
        self.agglo_groupbox_VBoxLayout.addWidget(self.elapsed_time_agglomerative)

        # Create a GroupBox and add K-Means things inside it
        self.k_means_groupbox = QGroupBox()
        self.k_means_groupbox.setTitle("K-Means")
        self.k_means_groupbox.setObjectName("k_means_groupbox")
        self.k_means_groupbox_VBoxLayout = QVBoxLayout()
        self.k_means_groupbox_VBoxLayout.setObjectName("k_means_groupbox_VBoxLayout")
        self.k_means_groupbox.setLayout(self.k_means_groupbox_VBoxLayout)
        self.k_means_groupbox_VBoxLayout.addLayout(self.n_clusters_HBoxLayout)
        self.k_means_groupbox_VBoxLayout.addLayout(
            self.k_means_max_iterations_HBoxLayout
        )
        self.k_means_groupbox_VBoxLayout.addWidget(self.spatial_segmentation)
        self.k_means_groupbox_VBoxLayout.addLayout(
            self.spatial_segmentation_weight_HBoxLayout
        )
        self.k_means_groupbox_VBoxLayout.addWidget(self.centroid_optimization)
        self.k_means_groupbox_VBoxLayout.addWidget(self.k_means_LUV_conversion)
        self.k_means_groupbox_VBoxLayout.addWidget(self.apply_k_means)

        # Create a GroupBox and add Mean Shift things inside it
        self.mean_shift_groupbox = QGroupBox()
        self.mean_shift_groupbox.setTitle("Mean Shift")
        self.mean_shift_groupbox.setObjectName("mean_shift_groupbox")
        self.mean_shift_groupbox_VBoxLayout = QVBoxLayout()
        self.mean_shift_groupbox_VBoxLayout.setObjectName(
            "mean_shift_groupbox_VBoxLayout"
        )
        self.mean_shift_groupbox.setLayout(self.mean_shift_groupbox_VBoxLayout)
        self.mean_shift_groupbox_VBoxLayout.addLayout(
            self.mean_shift_window_size_HBoxLayout
        )
        self.mean_shift_groupbox_VBoxLayout.addLayout(self.mean_shift_sigma_HBoxLayout)
        self.mean_shift_groupbox_VBoxLayout.addLayout(
            self.mean_shift_threshold_HBoxLayout
        )
        self.mean_shift_groupbox_VBoxLayout.addWidget(self.mean_shift_LUV_conversion)
        self.mean_shift_groupbox_VBoxLayout.addWidget(self.apply_mean_shift)

        # Add GroupBoxes to the main layout
        self.main_layout.addWidget(self.rg_groupbox)
        self.main_layout.addWidget(self.agglo_groupbox)
        self.main_layout.addWidget(self.k_means_groupbox)
        self.main_layout.addWidget(self.mean_shift_groupbox)

        self.setLayout(self.main_layout)
