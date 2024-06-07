import os

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import webbrowser

import matplotlib.pyplot as plt

# in CMD: pip install qdarkstyle -> pip install pyqtdarktheme
import qdarktheme
from Classes.ExtendedWidgets.CustomTabWidget import CustomTabWidget
from Classes.ExtendedWidgets.UserGuide import UserGuideDialog
from ImageAlchemyBackend import Backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QCursor, QIcon


class Ui_ImgProcessor(object):
    def setupUi(self, ImgAlchemy):

        # Main App Setup #
        # ============== #
        ImgAlchemy.setObjectName("ImgAlchemy")
        ImgAlchemy.resize(1280, 720)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("Resources/Icons/App_Icon.png"),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )
        ImgAlchemy.setWindowIcon(icon)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(11)
        ImgAlchemy.setFont(font)
        ImgAlchemy.setAcceptDrops(True)
        self.centralwidget = QtWidgets.QWidget(ImgAlchemy)
        self.centralwidget.setObjectName("centralwidget")
        ImgAlchemy.setCentralWidget(self.centralwidget)
        self.app_grid_layout = QtWidgets.QGridLayout(self.centralwidget)
        self.app_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.app_grid_layout.setSpacing(0)
        self.app_grid_layout.setObjectName("gridLayout")

        # Side Bar: Left Sidebar -> Closed by default #
        # =========================================== #

        # Header
        self.left_bar_header = QtWidgets.QFrame(self.centralwidget)
        self.left_bar_header.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.left_bar_header.setFrameShadow(QtWidgets.QFrame.Raised)
        self.left_bar_header.setObjectName("left_bar_header")
        # Header Layout
        self.left_bar_header_HLayout = QtWidgets.QHBoxLayout(self.left_bar_header)
        self.left_bar_header_HLayout.setObjectName("left_bar_header_HLayout")
        self.left_bar_header_HLayout.setContentsMargins(0, 0, 0, 0)
        # Header Buttonf
        self.left_bar_header_btn = QtWidgets.QPushButton(self.left_bar_header)
        self.left_bar_header_btn.setObjectName("left_bar_header_btn")
        self.left_bar_header_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.left_bar_header_HLayout.addWidget(self.left_bar_header_btn)
        # Header Button Properties
        self.left_bar_header_btn.setIcon(QIcon("Resources/Icons/RightArrows.png"))
        self.left_bar_header_btn.setIconSize(QSize(9, 9))
        self.left_bar_header_btn.setCheckable(True)
        self.left_bar_header_btn.setChecked(False)
        # Collapsed View
        self.left_bar_collapsed = QtWidgets.QWidget(self.centralwidget)
        self.left_bar_collapsed.setObjectName("left_bar_collapsed")
        self.left_bar_collapsed_VLayout = QtWidgets.QVBoxLayout(self.left_bar_collapsed)
        # Expanded View
        self.left_bar_expanded = QtWidgets.QWidget(self.centralwidget)
        self.left_bar_expanded.setObjectName("left_bar_expanded")
        self.left_bar_expanded_VLayout = QtWidgets.QVBoxLayout(self.left_bar_expanded)

        # Viewport: Image Workspace #
        # ========================= #

        # Main Tab Widget
        self.image_workspace = CustomTabWidget(self.centralwidget)
        self.image_workspace.setObjectName("Image_workspace")
        # Main Viewport Page (First tab of the tab widget)
        self.main_viewport = QtWidgets.QWidget()
        self.main_viewport.setObjectName("main_viewport")
        # Main Viewport Layout
        self.main_viewport_grid_layout = QtWidgets.QGridLayout(self.main_viewport)
        self.main_viewport_grid_layout.setObjectName("gridLayout_2")
        # Set the margins and padding of the layout to zero
        self.main_viewport_grid_layout.setContentsMargins(0, 0, 0, 0)
        # Main Viewport Frame
        self.main_viewport_frame = QtWidgets.QFrame(self.main_viewport)
        self.main_viewport_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_viewport_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_viewport_frame.setObjectName("main_viewport_canvas")
        # Start Main Viewport Canvas
        self.main_viewport_canvas_HLayout = QtWidgets.QVBoxLayout(
            self.main_viewport_frame
        )
        self.main_viewport_canvas_HLayout.setObjectName("main_viewport_canvas_HLayout")
        self.main_viewport_canvas_HLayout.setContentsMargins(0, 0, 0, 0)
        self.main_viewport_figure = plt.figure()
        self.main_viewport_figure_canvas = FigureCanvas(self.main_viewport_figure)
        # Change the background color of the canvas
        self.main_viewport_figure_canvas.figure.patch.set_facecolor("none")
        # Add the canvas to the layout
        self.main_viewport_canvas_HLayout.addWidget(self.main_viewport_figure_canvas)
        # Add the canvas to the main viewport layout
        self.main_viewport_grid_layout.addWidget(self.main_viewport_frame, 0, 0, 1, 1)
        # Add the tab to the tab widget
        self.image_workspace.addTab(self.main_viewport, "")
        # Remove the close button from the first tab to make it unclosable
        self.image_workspace.tabBar().setTabButton(0, QtWidgets.QTabBar.RightSide, None)

        # Effect Side Menu to edit their parameters: Closed by default #
        # ============================================================ #

        # Header
        self.effect_menu_header = QtWidgets.QFrame(self.centralwidget)
        self.effect_menu_header.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.effect_menu_header.setFrameShadow(QtWidgets.QFrame.Raised)
        self.effect_menu_header.setObjectName("effect_menu_header")
        # Header Layout
        self.effect_menu_header_HLayout = QtWidgets.QHBoxLayout(self.effect_menu_header)
        self.effect_menu_header_HLayout.setObjectName("effect_menu_header_HLayout")
        self.effect_menu_header_HLayout.setContentsMargins(0, 0, 0, 0)
        # Header Button
        self.effect_menu_header_btn = QtWidgets.QPushButton(self.effect_menu_header)
        self.effect_menu_header_btn.setObjectName("effect_menu_header_btn")
        self.effect_menu_header_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.effect_menu_header_HLayout.addWidget(self.effect_menu_header_btn)
        # Header Button Properties
        self.effect_menu_header_btn.setIcon(QIcon("Resources/Icons/LeftArrows.png"))
        self.effect_menu_header_btn.setIconSize(QSize(9, 9))
        self.effect_menu_header_btn.setCheckable(True)
        self.effect_menu_header_btn.setChecked(False)
        # Collapsed View
        self.effect_menu_collapsed = QtWidgets.QWidget(self.centralwidget)
        self.effect_menu_collapsed.setObjectName("effect_menu_collapsed")
        self.effect_menu_collapsed.setMinimumWidth(30)
        self.effect_menu_collapsed_VLayout = QtWidgets.QVBoxLayout(
            self.effect_menu_collapsed
        )
        # Expanded View
        self.effect_menu_expanded = QtWidgets.QWidget(self.centralwidget)
        self.effect_menu_expanded.setObjectName("effect_menu_expanded")
        self.effect_menu_expanded.setMaximumWidth(240)
        self.effect_menu_expanded_VLayout = QtWidgets.QVBoxLayout(
            self.effect_menu_expanded
        )
        self.effect_menu_expanded_VLayout.setContentsMargins(0, 0, 0, 0)
        self.effect_menu_expanded.setMinimumWidth(200)

        # Effect Menu: Expanded View Content #
        # ================================== #
        # Create a scroll area
        self.effect_menu_scroll_area = QtWidgets.QScrollArea()
        self.effect_menu_scroll_area.setWidgetResizable(True)

        # Add the scroll area to the expanded view
        self.effect_menu_expanded_VLayout.addWidget(self.effect_menu_scroll_area)

        # Scrol area content
        self.effect_menu_scroll_area_content = QtWidgets.QWidget()
        self.effect_menu_scroll_area_content.setObjectName(
            "effect_menu_scroll_area_content"
        )
        self.scroll_area_VLayout = QtWidgets.QVBoxLayout(
            self.effect_menu_scroll_area_content
        )
        self.scroll_area_VLayout.setObjectName("scroll_area_VLayout")
        self.effect_menu_scroll_area.setWidget(self.effect_menu_scroll_area_content)
        self.add_vertical_spacer(self.scroll_area_VLayout)

        # Control Panel and History: Opened by default #
        # ============================================ #

        # Header
        self.control_panel_header = QtWidgets.QFrame(self.centralwidget)
        self.control_panel_header.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.control_panel_header.setFrameShadow(QtWidgets.QFrame.Raised)
        self.control_panel_header.setObjectName("control_panel_header")
        # Header Layout
        self.control_panel_header_HLayout = QtWidgets.QHBoxLayout(
            self.control_panel_header
        )
        self.control_panel_header_HLayout.setObjectName("control_panel_header_HLayout")
        self.control_panel_header_HLayout.setContentsMargins(0, 0, 0, 0)
        # Header Button
        self.control_panel_header_btn = QtWidgets.QPushButton(self.control_panel_header)
        self.control_panel_header_btn.setObjectName("control_panel_header_btn")
        self.control_panel_header_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.control_panel_header_HLayout.addWidget(self.control_panel_header_btn)
        # Header Button Properties
        self.control_panel_header_btn.setIcon(QIcon("Resources/Icons/RightArrows.png"))
        self.control_panel_header_btn.setIconSize(QSize(9, 9))
        self.control_panel_header_btn.setCheckable(True)
        self.control_panel_header_btn.setChecked(False)
        # Collapsed View
        self.control_panel_collapsed = QtWidgets.QWidget(self.centralwidget)
        self.control_panel_collapsed.setObjectName("control_panel_collapsed")
        self.control_panel_collapsed.setMaximumWidth(340)
        self.control_panel_collapsed.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        self.control_panel_collapsed_VLayout = QtWidgets.QVBoxLayout(
            self.control_panel_collapsed
        )
        # Expanded View
        self.control_panel_expanded = QtWidgets.QWidget(self.centralwidget)
        self.control_panel_expanded.setObjectName("control_panel_expanded")
        self.control_panel_expanded.setMaximumWidth(340)
        self.control_panel_expanded.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        self.control_panel_expanded_VLayout = QtWidgets.QVBoxLayout(
            self.control_panel_expanded
        )
        self.control_panel_expanded_VLayout.setContentsMargins(0, 0, 0, 0)

        # Control Panel Collapsed Content #
        # =============================== #

        # Image History Tab Button
        self.img_history_tab_btn = QtWidgets.QPushButton(self.control_panel_collapsed)
        self.img_history_tab_btn.setObjectName("img_history_tab_btn")
        self.img_history_tab_btn.setIcon(QIcon("Resources/Icons/history.png"))
        self.img_history_tab_btn.setIconSize(QSize(20, 20))
        self.img_history_tab_btn.setText("  Image History")
        self.img_history_tab_btn.clicked.connect(self.toggle_CP_using_panels_icons)
        self.img_history_tab_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.control_panel_collapsed_VLayout.addWidget(self.img_history_tab_btn)
        # Added Effects Tab Button
        self.added_effects_tab_btn = QtWidgets.QPushButton(self.control_panel_collapsed)
        self.added_effects_tab_btn.setObjectName("added_effects_tab_btn")
        self.added_effects_tab_btn.setIcon(QIcon("Resources/Icons/effects.png"))
        self.added_effects_tab_btn.setIconSize(QSize(20, 20))
        self.added_effects_tab_btn.setText("  Added Effects")
        self.added_effects_tab_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.added_effects_tab_btn.clicked.connect(self.toggle_CP_using_panels_icons)
        self.control_panel_collapsed_VLayout.addWidget(self.added_effects_tab_btn)
        # Vertical Spacer
        self.add_vertical_spacer(self.control_panel_collapsed_VLayout)

        # Control Panel Expanded Content #
        # ============================== #

        # Image History Tab Widget
        self.img_history_tab_widget = QtWidgets.QTabWidget(self.control_panel_expanded)
        self.img_history_tab_widget.setObjectName("img_history_tab_widget")
        self.img_history_tab = QtWidgets.QWidget()
        self.img_history_tab.setObjectName("img_history_tab")
        self.img_history_tab_widget.addTab(self.img_history_tab, "")

        self.img_history_tab_layout = QtWidgets.QVBoxLayout(self.img_history_tab)
        self.img_history_tab_layout.setObjectName("img_history_tab_layout")
        self.img_history_tab_layout.setContentsMargins(0, 0, 0, 0)

        # Image History Tab Widget Content
        # Tool Bar inside the added effects menu
        self.image_history_tool_bar = QtWidgets.QToolBar(self.img_history_tab)
        self.image_history_tool_bar.setObjectName("added_effects_tool_bar")
        self.image_history_tool_bar.setMovable(False)
        self.image_history_tool_bar.setIconSize(QSize(18, 18))
        self.img_history_tab_layout.addWidget(self.image_history_tool_bar)

        # clear all effects in tool bar
        self.clear_history_btn = QtWidgets.QToolButton(self.img_history_tab)
        self.clear_history_btn.setObjectName("clear_history_btn")
        self.clear_history_btn.setIcon(QIcon("Resources/Icons/clear.png"))
        self.clear_history_btn.setToolTip(
            "Clear history\nBe Aware: Every thing will be reset!"
        )
        self.clear_history_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.image_history_tool_bar.addWidget(self.clear_history_btn)

        # Add a tree widget with alternating row colors
        self.img_history_tree = QtWidgets.QTreeWidget(self.img_history_tab)
        self.img_history_tree.setHeaderHidden(True)
        self.img_history_tree.setAlternatingRowColors(True)
        self.img_history_tree.setExpandsOnDoubleClick(False)
        self.img_history_tab_layout.addWidget(self.img_history_tree)

        # Add the image history and added effects section to the expanded view
        self.control_panel_expanded_VLayout.addWidget(self.img_history_tab_widget)

        # MenuBar: File || View || Tools || Help #
        # ====================================== #
        # This is only one that its variables are written in camel case (menuFile)

        # MenuBar: Setup
        self.menubar = QtWidgets.QMenuBar(ImgAlchemy)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 21))
        self.menubar.setObjectName("menubar")
        ImgAlchemy.setMenuBar(self.menubar)
        # MenuBar: Menues
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuLibrary")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")

        # Add Menus to menubar
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        # MenuBar -> menuFile actions
        # Import Image
        self.actionImport_Image = QtWidgets.QAction(ImgAlchemy)
        self.actionImport_Image.setObjectName("actionImport_Image")
        self.actionImport_Image.setShortcut("Ctrl+I")
        # Save current
        self.actionSave_current = QtWidgets.QAction(ImgAlchemy)
        self.actionSave_current.setObjectName("actionSave_current")
        self.actionSave_current.setShortcut("Ctrl+S")
        # Save current image as
        self.actionSave_as = QtWidgets.QAction(ImgAlchemy)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionSave_as.setShortcut("Shift+Ctrl+S")
        # Save all opened and edited images
        self.actionSave_all = QtWidgets.QAction(ImgAlchemy)
        self.actionSave_all.setObjectName("actionSave_all")
        self.actionSave_all.setShortcut("Alt+S")
        # Exit the app
        self.actionExit = QtWidgets.QAction(ImgAlchemy)
        self.actionExit.setObjectName("actionExit_2")
        self.actionExit.setShortcut("Ctrl+Q")

        # Add actions to menuFile
        self.menuFile.addAction(self.actionImport_Image)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_current)
        self.menuFile.addAction(self.actionSave_as)
        self.menuFile.addAction(self.actionSave_all)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)

        # MenuBar -> menuEdit actions
        # menuEdit submenus
        self.actionThemes = QtWidgets.QAction(self.menuEdit)
        self.actionThemes.setObjectName("Themes")
        # self.actionThemes.triggered.connect(self.change_app_theme)
        # Add submenus to menuTools
        self.menuEdit.addAction(self.actionThemes)

        # MenuBar -> menuTools actions
        # menuTools submenus
        self.menu_standard_images_library = QtWidgets.QMenu(self.menuTools)
        self.menu_standard_images_library.setObjectName("menu_standard_images_library")
        self.menu_hough_samples = QtWidgets.QMenu(self.menuTools)
        self.menu_hough_samples.setObjectName("menu_hough_samples")
        # Add submenus to menuTools
        self.menuTools.addAction(self.menu_standard_images_library.menuAction())
        self.menuTools.addAction(self.menu_hough_samples.menuAction())

        # Actions for the standard library
        self.actionClassic = QtWidgets.QAction(ImgAlchemy)
        self.actionClassic.setObjectName("actionClassic")
        self.actionOld_Classic = QtWidgets.QAction(ImgAlchemy)
        self.actionOld_Classic.setObjectName("actionOld_Classic")
        self.actionMedical = QtWidgets.QAction(ImgAlchemy)
        self.actionMedical.setObjectName("actionColored")
        self.actionSun_and_Plants = QtWidgets.QAction(ImgAlchemy)
        self.actionSun_and_Plants.setObjectName("actionGrayscale")
        self.actionHigh_Resolution = QtWidgets.QAction(ImgAlchemy)
        self.actionHigh_Resolution.setObjectName("actionHigh_Resolution")
        self.actionFingerprints = QtWidgets.QAction(ImgAlchemy)
        self.actionFingerprints.setObjectName("actionFingerprints")
        self.actionTextures = QtWidgets.QAction(ImgAlchemy)
        self.actionTextures.setObjectName("actionTextures")
        self.actionSpecial = QtWidgets.QAction(ImgAlchemy)
        self.actionSpecial.setObjectName("actionSpecial")
        self.actionAdditional = QtWidgets.QAction(ImgAlchemy)
        self.actionAdditional.setObjectName("actionAdditional")

        # Actions for the hough samples
        self.actionLine = QtWidgets.QAction(ImgAlchemy)
        self.actionLine.setObjectName("actionLine")
        self.actionCircle = QtWidgets.QAction(ImgAlchemy)
        self.actionCircle.setObjectName("actionCircle")
        self.actionEllipse = QtWidgets.QAction(ImgAlchemy)
        self.actionEllipse.setObjectName("actionEllipse")
        self.menu_hough_samples.addAction(self.actionLine)
        self.menu_hough_samples.addAction(self.actionCircle)
        self.menu_hough_samples.addAction(self.actionEllipse)

        # Add actions to menuTools/submenu
        self.menu_standard_images_library.addAction(self.actionClassic)
        self.menu_standard_images_library.addAction(self.actionOld_Classic)
        self.menu_standard_images_library.addAction(self.actionMedical)
        self.menu_standard_images_library.addAction(self.actionSun_and_Plants)
        self.menu_standard_images_library.addAction(self.actionHigh_Resolution)
        self.menu_standard_images_library.addAction(self.actionFingerprints)
        self.menu_standard_images_library.addAction(self.actionTextures)
        self.menu_standard_images_library.addAction(self.actionSpecial)
        self.menu_standard_images_library.addAction(self.actionAdditional)

        # MenuBar -> menuHelp actions
        # Controls
        self.actionControls = QtWidgets.QAction(ImgAlchemy)
        self.actionControls.setObjectName("actionControls")
        self.actionControls.setShortcut("Ctrl+C")
        self.actionControls.triggered.connect(self.open_user_guide)
        # App Documentation
        self.actionApp_Documentation = QtWidgets.QAction(ImgAlchemy)
        self.actionApp_Documentation.setObjectName("actionApp_Documentation")
        self.actionApp_Documentation.setShortcut("Ctrl+H")
        self.actionApp_Documentation.triggered.connect(self.open_documentation)

        # Add actions to menuHelp
        self.menuHelp.addAction(self.actionControls)
        self.menuHelp.addAction(self.actionApp_Documentation)

        # StatusBar: Setup #
        # ================ #
        self.statusbar = QtWidgets.QStatusBar(ImgAlchemy)
        self.statusbar.setObjectName("statusbar")
        ImgAlchemy.setStatusBar(self.statusbar)

        # Initial conditions of different side bar #
        # ======================================== #
        # Left SideBar
        self.left_bar_expanded.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.left_bar_collapsed.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.left_bar_expanded.hide()  # Hide the ExpandedView initially
        self.left_bar_collapsed.show()  # Show the CollapsedView initially
        # Effect Menu
        self.effect_menu_expanded.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.effect_menu_collapsed.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.effect_menu_expanded.hide()  # Hide the ExpandedView initially
        self.effect_menu_collapsed.show()  # Show the CollapsedView initially
        # Control Panel
        self.control_panel_expanded.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.control_panel_collapsed.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.control_panel_expanded.show()  # Show the CollapsedView initially
        self.control_panel_collapsed.hide()  # Hide the ExpandedView initially

        # Manage of the app main layout #
        # ============================= #

        # Before adding the resizability feature
        self.app_grid_layout.addWidget(self.left_bar_header, 0, 0, 1, 2)
        self.app_grid_layout.addWidget(self.left_bar_collapsed, 1, 0, 1, 1)
        self.app_grid_layout.addWidget(self.left_bar_expanded, 1, 1, 1, 1)

        self.app_grid_layout.addWidget(self.image_workspace, 0, 2, 2, 1)

        self.app_grid_layout.addWidget(self.effect_menu_header, 0, 3, 1, 2)
        self.app_grid_layout.addWidget(self.effect_menu_collapsed, 1, 3, 1, 1)
        self.app_grid_layout.addWidget(self.effect_menu_expanded, 1, 4, 1, 1)

        self.app_grid_layout.addWidget(self.control_panel_header, 0, 5, 1, 2)
        self.app_grid_layout.addWidget(self.control_panel_collapsed, 1, 5, 1, 1)
        self.app_grid_layout.addWidget(self.control_panel_expanded, 1, 6, 1, 1)

        # Instantiate the user guide once #
        # =============================== #

        self.init_bars_signals_and_slots()
        self.retranslateUi(ImgAlchemy)
        self.image_workspace.setCurrentIndex(0)
        self.img_history_tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(ImgAlchemy)

    def add_vertical_spacer(self, layout):
        self.vertical_spacer = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        layout.addItem(self.vertical_spacer)

    def init_bars_signals_and_slots(self):
        # connect signals and slots for the menu button and the side menu
        self.left_bar_header_btn.toggled["bool"].connect(
            self.left_bar_expanded.setHidden
        )
        self.left_bar_header_btn.toggled["bool"].connect(
            self.left_bar_collapsed.setVisible
        )

        # connect signals and slots for the menu button and the effect edit menu
        self.effect_menu_header_btn.toggled["bool"].connect(
            self.effect_menu_expanded.setHidden
        )
        self.effect_menu_header_btn.toggled["bool"].connect(
            self.effect_menu_collapsed.setVisible
        )

        # connect signals and slots for the menu button and the effect edit menu
        self.control_panel_header_btn.toggled["bool"].connect(
            self.control_panel_expanded.setHidden
        )
        self.control_panel_header_btn.toggled["bool"].connect(
            self.control_panel_collapsed.setVisible
        )

        # change the SideBar button appearance
        self.left_bar_header_btn.toggled.connect(self.toggle_left_side_bar)
        self.effect_menu_header_btn.toggled.connect(self.toggle_effect_bar)
        self.control_panel_header_btn.toggled.connect(self.toggle_control_panel)

    def toggle_left_side_bar(self, state):
        if state:
            # If the button is checked (expanded view is shown), hide the CollapsedView and show the ExpandedView
            self.left_bar_collapsed.hide()
            self.left_bar_expanded.show()
            self.left_bar_header_btn.setIcon(QIcon("Resources/Icons/LeftArrows.png"))
        else:
            # If the button is unchecked (collapsed view is shown), hide the ExpandedView and show the CollapsedView
            self.left_bar_expanded.hide()
            self.left_bar_collapsed.show()
            self.left_bar_header_btn.setIcon(QIcon("Resources/Icons/RightArrows.png"))

    def toggle_control_panel(self, state):
        if state:
            # If the button is checked (expanded view is shown), hide the CollapsedView and show the ExpandedView
            self.control_panel_expanded.hide()
            self.control_panel_collapsed.show()
            self.control_panel_header_btn.setIcon(
                QIcon("Resources/Icons/LeftArrows.png")
            )
        else:
            # If the button is unchecked (collapsed view is shown), hide the ExpandedView and show the CollapsedView
            self.control_panel_collapsed.hide()
            self.control_panel_expanded.show()
            self.control_panel_header_btn.setIcon(
                QIcon("Resources/Icons/RightArrows.png")
            )

    def toggle_effect_bar(self, state):
        if state:
            # If the button is checked (expanded view is shown), hide the CollapsedView and show the ExpandedView
            self.effect_menu_collapsed.hide()
            self.effect_menu_expanded.show()
            self.effect_menu_header_btn.setIcon(
                QIcon("Resources/Icons/RightArrows.png")
            )
        else:
            # If the button is unchecked (collapsed view is shown), hide the ExpandedView and show the CollapsedView
            self.effect_menu_expanded.hide()
            self.effect_menu_collapsed.show()
            self.effect_menu_header_btn.setIcon(QIcon("Resources/Icons/LeftArrows.png"))

    def toggle_CP_using_panels_icons(self):
        self.control_panel_header_btn.setChecked(
            not self.control_panel_header_btn.isChecked()
        )

    def open_user_guide(self):
        user_guide = UserGuideDialog()
        user_guide.exec_()

    def open_documentation(self):
        webbrowser.open("https://github.com/Computer-Vision-Spring-2024/Task-1")

    def retranslateUi(self, ImgProcessor):
        _translate = QtCore.QCoreApplication.translate
        ImgProcessor.setWindowTitle(_translate("ImgProcessor", "Image Alchemy"))
        self.left_bar_header_btn.setText(_translate("ImgProcessor", ""))
        self.image_workspace.setTabText(
            self.image_workspace.indexOf(self.main_viewport),
            _translate("ImgProcessor", "Main Viewport"),
        )
        self.menuFile.setTitle(_translate("ImgProcessor", "File"))
        self.menuEdit.setTitle(_translate("ImgProcessor", "Edit"))
        self.menuHelp.setTitle(_translate("ImgProcessor", "Help"))
        self.menuTools.setTitle(_translate("ImgProcessor", "Tools"))
        self.menu_standard_images_library.setTitle(
            _translate("ImgProcessor", "Standard Images Library")
        )
        self.menu_hough_samples.setTitle(_translate("ImgProcessor", "Hough Samples"))
        self.img_history_tab_widget.setTabText(
            self.img_history_tab_widget.indexOf(self.img_history_tab),
            _translate("ImgProcessor", "Image History"),
        )
        self.actionControls.setText(_translate("ImgProcessor", "User Guide"))
        self.actionApp_Documentation.setText(
            _translate("ImgProcessor", "App Documentation")
        )
        self.actionImport_Image.setText(_translate("ImgProcessor", "Import Image"))
        self.actionSave_current.setText(_translate("ImgProcessor", "Save Current"))
        self.actionMedical.setText(_translate("ImgProcessor", "Medical"))
        self.actionSun_and_Plants.setText(_translate("ImgProcessor", "Sun and Planets"))
        self.actionSave_as.setText(_translate("ImgProcessor", "Save as"))
        self.actionSave_all.setText(_translate("ImgProcessor", "Save All"))
        self.actionExit.setText(_translate("ImgProcessor", "Exit"))
        self.actionClassic.setText(_translate("ImgProcessor", "Classic"))
        self.actionOld_Classic.setText(_translate("ImgProcessor", "Old Classic"))
        self.actionHigh_Resolution.setText(
            _translate("ImgProcessor", "High Resolution")
        )
        self.actionAdditional.setText(_translate("ImgProcessor", "Additional"))
        self.actionSpecial.setText(_translate("ImgProcessor", "Special"))
        self.actionFingerprints.setText(_translate("ImgProcessor", "Fingerprints"))
        self.actionTextures.setText(_translate("ImgProcessor", "Textures"))
        self.actionLine.setText(_translate("ImgProcessor", "Line"))
        self.actionCircle.setText(_translate("ImgProcessor", "Circle"))
        self.actionEllipse.setText(_translate("ImgProcessor", "Ellipse"))
        self.actionThemes.setText(_translate("ImgProcessor", "Themes"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # with open("Resources\Themes\BlackTheme.qss", "r") as f:
    #     stylesheet = f.read()
    #     app.setStyleSheet(stylesheet)
    ImgProcessor = QtWidgets.QMainWindow()
    ui = Ui_ImgProcessor()
    ui.setupUi(ImgProcessor)
    backend = Backend(ui)
    ImgProcessor.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())
