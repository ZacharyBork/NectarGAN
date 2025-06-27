from typing import Literal
from importlib.resources import files

from PySide6.QtCore import (
    Qt, QSize, QEvent, QObject, QPropertyAnimation,
    QVariantAnimation, QEasingCurve)
from PySide6.QtGui import QPixmap, QTransform, QMovie
from PySide6.QtWidgets import (
    QWidget, QDockWidget, QLabel, QFrame, QPushButton, 
    QDockWidget, QStackedWidget,)

import nectargan.toolbox.utils.common as utils

class SettingsDock(QObject):
    def __init__(self, mainwidget: QWidget):
        super().__init__()
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild
        self._get_widgets()

        self.button_names = [
            'experiment', 'dataset', 'training', 'testing', 'review', 'utils', 'settings']
        self.button_icons = [
            'flask', 'database', 'barbell', 'test-tube', 'clipboard', 'wrench', 'gear']

    ### INIT HELPERS ###

    def _get_widgets(self) -> None:
        self.dock = self.find(QDockWidget, 'settings_dock')
        self.central_pages = self.find(QStackedWidget, 'centralwidget_pages')
        self.main_icon = self.find(QLabel, 'settings_dock_main_icon')
        self.section_header = self.find(QLabel, 'settings_section_header')
        self.section_icon = self.find(QLabel, 'settings_section_header_icon')
        self.btns_frame = self.find(QFrame, 'settings_btns_frame')
        self.title_tag = self.find(QLabel, 'title_tag')
        self.creator_tag = self.find(QLabel, 'creator_tag')

    def _init_training_animation(self) -> None:
        animation_filepath = files(
            'nectargan.toolbox.resources.gifs').joinpath('train_lock.gif')

        self.train_lock_anim = QMovie(str(animation_filepath))
        self.train_lock_label = self.find(QLabel, 'train_page_locked_label')
        self.train_lock_label.setMovie(self.train_lock_anim)

        width = self.find(QWidget, 'train_page_state_locked').width() * 0.5
        self.train_lock_anim.setScaledSize(QSize(width, width*2))
        self.train_lock_anim.setCacheMode(QMovie.CacheMode.CacheAll)
        self.train_lock_anim.setSpeed(100)

    ### STATES ###

    def _set_navbutton_state(self, name: str, enabled: bool) -> None:
        button = self.find(QPushButton, f'{name}_settings_btn')
        label = self.find(QPushButton, f'{name}_settings_label')

        icon_name = self.button_icons[self.button_names.index(name)]
        if not enabled: icon_name = f'{icon_name}_gray'
        utils.set_button_icon(f'{icon_name}.svg', button)
        button.setEnabled(enabled)
        label.setEnabled(enabled)

    def _set_all_navbutton_states(self, state: bool) -> None:
        for name in self.button_names:
            self._set_navbutton_state(name, state)

    def set_state(self, state: Literal['init', 'training']) -> None:
        match state:
            case 'init':
                self._set_all_navbutton_states(True)
                self.train_lock_anim.stop()
            case 'training':
                self._set_navbutton_state('experiment', False)
                self._set_navbutton_state('dataset', False)
                self._set_navbutton_state('testing', False)
                self._set_navbutton_state('review', False)
                self._set_navbutton_state('utils', False)
                self.train_lock_anim.start()

    ### CALLBACKS ###

    def _set_section_icons(self, index: int=0) -> None:
        file = utils.get_icon_file(f'{self.button_icons[index]}.svg')
        transform = QTransform()
        pixmap = QPixmap(file.as_posix()).transformed(transform)
        self.section_icon.setPixmap(pixmap)

    def _change_page(self, index: int) -> None:
        self.section_header.setText(self.button_names[index].capitalize())
        self._set_section_icons(index=index)

        for name in self.button_names:
            button = self.find(QPushButton, f'{name}_settings_btn')
            if not name == self.button_names[index]:
                button.setChecked(False)
            else: button.setChecked(True)
        stack = self.find(QStackedWidget, 'settings_pages')
        try: stack.setCurrentIndex(index)
        except: pass
        match index:
            case 2 | 3 | 4 | 5:
                self.central_pages.setCurrentIndex(index-2)
            case _: pass

    def _show_logo_animation(self) -> None:
        self.logo_anim = QMovie(str(files(
            'nectargan.toolbox.resources.gifs').joinpath('logo_anim.gif')))
        self.main_icon.setMovie(self.logo_anim)
        self.logo_anim.setCacheMode(QMovie.CacheMode.CacheAll)
        self.logo_anim.setSpeed(100)
        self.logo_anim.setScaledSize(QSize(165, 55))
        self.logo_anim.start()
        
        self.main_icon.setFixedSize(170, 55)

        self.logo_anim_timer = QVariantAnimation()
        self.logo_anim_timer.setStartValue(10000)
        self.logo_anim_timer.setEndValue(0)
        self.logo_anim_timer.setDuration(1100)
        self.logo_anim_timer.valueChanged.connect(
            lambda x : self._swap_animation(value=x))
        self.logo_anim_timer.start()
    
    def _swap_animation(self, value: int) -> None:
        if value <= 0:
            self.logo_anim.stop()
            self.logo_anim_timer.stop()
            self._set_main_icon(size=(220, 55), file='toolbox_icon_text.png')
            self.main_icon.setFixedSize(170, 55)
            
    def _set_main_icon(
            self, 
            size: tuple[int],
            file: str='toolbox_icon.png'
        ) -> None:
        try: self.logo_anim_timer.stop()
        except: pass
        main_icon_pixmap = QPixmap(
            utils.get_icon_file(file)).scaled(
                size[0], size[1],  Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation)
        self.main_icon.setPixmap(main_icon_pixmap)
        self.main_icon.setFixedSize(size[0], size[1])

    def _swap_btn_label_visibility(
            self, 
            mode: Literal['show', 'hide']
        ) -> None:
        if mode == 'show':
            for i, name in enumerate(self.button_names):
                label = self.find(QPushButton, f'{name}_settings_label')
                label.setHidden(False)
        elif mode == 'hide': 
            for i, name in enumerate(self.button_names):
                label = self.find(QPushButton, f'{name}_settings_label')
                label.setHidden(True)
    
    def _init_navbar_expansion(self) -> None:
        self.btns_frame.installEventFilter(self)

    def _animate_nav_panel(self, expand: bool):
        width = 180 if expand else 65
        self.nav_anim = QPropertyAnimation(self.btns_frame, b'maximumWidth')
        self.nav_anim.setDuration(300)
        self.nav_anim.setStartValue(self.btns_frame.width())
        self.nav_anim.setEndValue(width)
        self.nav_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.nav_anim.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched == getattr(self, 'btns_frame', None):
            if event.type() == QEvent.Type.Enter:
                self._animate_nav_panel(expand=True)
                # self._set_main_icon(
                #    size=(220, 55), file='toolbox_icon_text.png')
                self._show_logo_animation()
                # self.main_icon.setFixedSize(170, 55)
                self._swap_btn_label_visibility('show')
                self.title_tag.setText('NectarGAN Toolbox')
                self.creator_tag.setText('Created by Zachary Bork')
            elif event.type() == QEvent.Type.Leave:
                self._animate_nav_panel(expand=False)
                self._set_main_icon(size=(55, 55))
                self._swap_btn_label_visibility('hide')
                self.title_tag.setText('')
                self.creator_tag.setText('')
        return super().eventFilter(watched, event)

    def init(self) -> None:
        self.dock.setTitleBarWidget(QWidget(None))
        self.title_tag.setText('')
        self.creator_tag.setText('')
        self._set_main_icon(size=(55, 55))

        self.btns_frame.setMouseTracking(True)
        self._init_navbar_expansion()
        self._set_section_icons()

        self._init_training_animation()
        self.set_state('init')

        for i, name in enumerate(self.button_names):
            button = self.find(QPushButton, f'{name}_settings_btn')
            button.clicked.connect(lambda x, i=i : self._change_page(i))
            utils.set_button_icon(f'{self.button_icons[i]}.svg', button)

            label = self.find(QPushButton, f'{name}_settings_label')
            label.clicked.connect(lambda x, i=i : self._change_page(i))
            label.setHidden(True)

