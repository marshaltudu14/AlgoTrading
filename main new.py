import requests
import base64
from datetime import datetime, timedelta, date
from datetime import time as dt_time
import time
import threading
import pyotp
from pytz import timezone
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_ta as ta
import os
import pytz
import json
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, Bidirectional, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import Callback, ModelCheckpoint
from scipy.signal import argrelextrema
import tensorflow as tf

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock, mainthread
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.metrics import dp
from kivy.core.audio import SoundLoader
from kivy_garden.matplotlib import FigureCanvasKivyAgg

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

KV = '''
ScreenManager:
    AuthScreen:
    MenuScreen:
    BacktestScreen:
    FinalScreen:

<AuthScreen>:
    name: 'auth'
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            pos: self.pos
            size: self.size
    BoxLayout:
        orientation: 'vertical'
        spacing: dp(50)
        padding: dp(20)
        size_hint: None, None
        size: self.minimum_size
        pos_hint: {'center_x': 0.5, 'center_y': 0.5}
        Label:
            text: 'Fyers Algo Bot'
            color: 1, 1, 1, 1
            font_size: dp(50)
        BoxLayout:
            orientation: 'horizontal'
            spacing: dp(20)
            size_hint: None, None
            size: self.minimum_size
            Button:
                text: 'Start'
                size_hint: None, None
                size: dp(200), dp(50)
                background_normal: ''
                background_color: 0.95, 0.38, 0.25, 1
                color: 1, 1, 1, 1
                on_release: root.start_auth()
            Button:
                text: 'Modify Credentials'
                size_hint: None, None
                size: dp(200), dp(50)
                background_normal: ''
                background_color: 0.95, 0.38, 0.25, 1
                color: 1, 1, 1, 1
                on_release: root.modify_credentials()

<MenuScreen>:
    name: 'menu'
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(20)

        ScrollView:
            BoxLayout:
                id: index_timeframe_layout
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(20)

                # Real/Virtual Trade Selection
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(50)
                    Label:
                        text: 'Real or Paper Trade:'
                        color: 1, 1, 1, 1
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        size_hint_x: None
                        width: self.parent.width
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(120)
                    GridLayout:
                        cols: 2
                        spacing: dp(20)
                        ToggleButton:
                            id: toggle_real_trade
                            text: 'Real'
                            group: 'trade_type'
                            background_normal: ''
                            background_color: 0.1, 0.1, 0.1, 1
                            on_press: root.trade_type_selected(self, self.text)

                        ToggleButton:
                            id: toggle_paper_trade
                            text: 'Paper'
                            group: 'trade_type'
                            background_normal: ''
                            background_color: 0.1, 0.1, 0.1, 1
                            on_press: root.trade_type_selected(self, self.text)

                        Label:
                            id: label_real_capital
                            text: ''
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        
                        TextInput:
                            id: input_paper_capital
                            hint_text: 'Capital'
                            text: ''
                            multiline: False
                            input_filter: 'int'
                            foreground_color: 1, 1, 1, 1
                            background_color: 0.1, 0.1, 0.1, 1

                # Index selection
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(50)
                    Label:
                        text: 'Index:'
                        color: 1, 1, 1, 1
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        size_hint_x: None
                        width: self.parent.width
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(120)
                    GridLayout:
                        cols: 3
                        spacing: dp(20)
                        ToggleButton:
                            id: toggle_nifty
                            text: 'Nifty'
                            group: 'index'
                            background_normal: ''
                            background_color: 0.1, 0.1, 0.1, 1
                            on_press: root.index_selected(self, self.text)

                        ToggleButton:
                            id: toggle_finnity
                            text: 'Finnifty'
                            group: 'index'
                            background_normal: ''
                            background_color: 0.1, 0.1, 0.1, 1
                            on_press: root.index_selected(self, self.text)

                        ToggleButton:
                            id: toggle_bank_nifty
                            text: 'Bank Nifty'
                            group: 'index'
                            background_normal: ''
                            background_color: 0.1, 0.1, 0.1, 1
                            on_press: root.index_selected(self, self.text)

                        ToggleButton:
                            id: toggle_bankex
                            text: 'Bankex'
                            group: 'index'
                            background_normal: ''
                            background_color: 0.1, 0.1, 0.1, 1
                            on_press: root.index_selected(self, self.text)

                        ToggleButton:
                            id: toggle_sensex
                            text: 'Sensex'
                            group: 'index'
                            background_normal: ''
                            background_color: 0.1, 0.1, 0.1, 1
                            on_press: root.index_selected(self, self.text)

                # Time Frame selection
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(50)
                    Label:
                        text: 'Time Frame:'
                        color: 1, 1, 1, 1
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        size_hint_x: None
                        width: self.parent.width
                BoxLayout:
                    orientation: 'horizontal'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(50)
                    ToggleButton:
                        id: toggle_1min
                        text: '1 MIN'
                        group: 'timeframe'
                        background_normal: ''
                        background_color: 0.1, 0.1, 0.1, 1
                        on_press: root.timeframe_selected(self, self.text)

                    ToggleButton:
                        id: toggle_2min
                        text: '2 MIN'
                        group: 'timeframe'
                        background_normal: ''
                        background_color: 0.1, 0.1, 0.1, 1
                        on_press: root.timeframe_selected(self, self.text)

                    ToggleButton:
                        id: toggle_3min
                        text: '3 MIN'
                        group: 'timeframe'
                        background_normal: ''
                        background_color: 0.1, 0.1, 0.1, 1
                        on_press: root.timeframe_selected(self, self.text)

                    ToggleButton:
                        id: toggle_5min
                        text: '5 MIN'
                        group: 'timeframe'
                        background_normal: ''
                        background_color: 0.1, 0.1, 0.1, 1
                        on_press: root.timeframe_selected(self, self.text)

                    ToggleButton:
                        id: toggle_15min
                        text: '15 MIN'
                        group: 'timeframe'
                        background_normal: ''
                        background_color: 0.1, 0.1, 0.1, 1
                        on_press: root.timeframe_selected(self, self.text)
                
                # Quantity selection
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(50)
                    Label:
                        text: 'Quantity:'
                        color: 1, 1, 1, 1
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        size_hint_x: None
                        width: self.parent.width
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(50)
                    TextInput:
                        id: quantity_text_input
                        hint_text: 'Enter quantity'
                        text: ''
                        multiline: False
                        input_filter: 'int'
                        foreground_color: 1, 1, 1, 1
                        background_color: 0.1, 0.1, 0.1, 1
                    

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: dp(50)
            spacing: dp(20)
            Button:
                text: 'Proceed'
                background_normal: ''
                background_color: 0.95, 0.38, 0.25, 1
                color: 1, 1, 1, 1
                on_release: root.proceed()

            Button:
                text: 'Back'
                background_normal: ''
                background_color: 0.95, 0.38, 0.25, 1
                color: 1, 1, 1, 1
                on_release: root.manager.current = 'auth'


<BacktestScreen>:
    name: 'backtest'
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(20)

        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: dp(60)
            spacing: dp(10)
            GridLayout:
                cols: 1
                spacing: dp(10)
                Button:
                    id: button_backtest_next
                    text: 'Next'
                    background_normal: ''
                    background_color: 0.95, 0.38, 0.25, 1
                    color: 1, 1, 1, 1
                    disabled: False
                    on_release: root.manager.current = 'final'

        ScrollView:
            BoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(20)

                BoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(350)
                    spacing: dp(10)
                    GridLayout:
                        cols: 3
                        spacing: dp(10)
                        Label:
                            text: 'Profit / Loss:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Entry Type:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Points:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_profitloss
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_buysell
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_points
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                    GridLayout:
                        cols: 3
                        spacing: dp(10)
                        Label:
                            text: 'Win %:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Capital:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Loss %:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_win_pct
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_capital
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_loss_pct
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                    GridLayout:
                        cols: 2
                        spacing: dp(10)
                        Label:
                            text: 'Brokerage:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Quantity:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_brokerage
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_backtest_quantity
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size

        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: dp(60)
            spacing: dp(10)
            GridLayout:
                cols: 3
                spacing: dp(10)
                Label:
                    text: 'Days (Max 100 Days):'
                    color: 1, 1, 1, 1
                    halign: 'left'
                    valign: 'middle'
                    text_size: self.size
                Label:
                    text: 'Initial Capital:'
                    color: 1, 1, 1, 1
                    halign: 'left'
                    valign: 'middle'
                    text_size: self.size
                Label:
                    text: 'Brokerage:'
                    color: 1, 1, 1, 1
                    halign: 'left'
                    valign: 'middle'
                    text_size: self.size
                TextInput:
                    id: text_input_backtest_data
                    text: '5'
                    multiline: False
                    input_filter: 'int'
                    foreground_color: 1, 1, 1, 1
                    background_color: 0.1, 0.1, 0.1, 1
                TextInput:
                    id: text_input_backtest_initial_capital
                    text: '10000'
                    multiline: False
                    input_filter: 'int'
                    foreground_color: 1, 1, 1, 1
                    background_color: 0.1, 0.1, 0.1, 1
                TextInput:
                    id: text_input_backtest_brokerage
                    text: '100'
                    multiline: False
                    input_filter: 'int'
                    foreground_color: 1, 1, 1, 1
                    background_color: 0.1, 0.1, 0.1, 1

        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: dp(60)
            spacing: dp(10)
            GridLayout:
                cols: 2
                spacing: dp(10)
                Button:
                    id: button_backtest_start
                    text: 'Backtest'
                    background_normal: ''
                    background_color: 0.95, 0.38, 0.25, 1
                    color: 1, 1, 1, 1
                    disabled: False
                    on_release: root.start_backtest()
                Button:
                    id: button_backtest_menu
                    text: 'Menu'
                    background_normal: ''
                    background_color: 0.95, 0.38, 0.25, 1
                    color: 1, 1, 1, 1
                    disabled: False
                    on_release: root.manager.current = 'menu'

<FinalScreen>:
    name: 'final'
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(20)

        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: dp(50)
            spacing: dp(20)
            GridLayout:
                cols: 2
                spacing: dp(20)
                Button:
                    id: button_backtest_model
                    text: 'Backtest'
                    background_normal: ''
                    background_color: 0.95, 0.38, 0.25, 1
                    on_release: root.final_backtest_navigation()
                Button:
                    id: button_setting
                    text: 'Menu'
                    background_normal: ''
                    background_color: 0.95, 0.38, 0.25, 1
                    on_release: root.final_menu_navigation()
            
        ScrollView:
            BoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(20)

                BoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(60)
                    spacing: dp(10)
                    GridLayout:
                        cols: 3
                        spacing: dp(10)
                        Label:
                            text: 'Profit %:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Candle Pattern:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Loss %:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_overall_profit
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_candle_pattern
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_overall_loss
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                
                BoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(60)
                    spacing: dp(10)
                    GridLayout:
                        cols: 3
                        spacing: dp(10)
                        Label:
                            text: 'Points:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Profit/Loss:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            text: 'Capital:'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.35, 0.14, 0.09, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_points_captured
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_profit_loss
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                        Label:
                            id: label_capital
                            text: 'None'
                            color: 1, 1, 1, 1
                            canvas.before:
                                Color:
                                    rgba: 0.1, 0.1, 0.1, 1
                                Rectangle:
                                    pos: self.pos
                                    size: self.size
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(700)
                    BoxLayout:
                        id: final_charts_layout
                        orientation: 'vertical'
                BoxLayout:
                    orientation: 'vertical'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(700)
                    BoxLayout:
                        id: final_index_charts_layout
                        orientation: 'vertical'
        
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: dp(50)
            spacing: dp(20)
            GridLayout:
                cols: 2
                spacing: dp(20)
                Button:
                    id: button_manual_ce_entry
                    text: 'CE Entry'
                    background_normal: ''
                    background_color: 0.95, 0.38, 0.25, 1
                    on_release: root.final_manual_ce_entry()
                Button:
                    id: button_manual_pe_entry
                    text: 'PE Entry'
                    background_normal: ''
                    background_color: 0.95, 0.38, 0.25, 1
                    on_release: root.final_manual_pe_entry()

<CredentialsPopup>:
    title: 'Modify Credentials'
    size_hint: 0.5, 0.5
    auto_dismiss: False
    BoxLayout:
        orientation: 'vertical'
        padding: dp(10)
        spacing: dp(5)
        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'App ID'
                color: 1, 1, 1, 1
            TextInput:
                id: app_id
                multiline: False
                foreground_color: 1, 1, 1, 1
                background_color: 0.1, 0.1, 0.1, 1
        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Secret Key'
                color: 1, 1, 1, 1
            TextInput:
                id: secret_key
                multiline: False
                foreground_color: 1, 1, 1, 1
                background_color: 0.1, 0.1, 0.1, 1
        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Redirect URL'
                color: 1, 1, 1, 1
            TextInput:
                id: redirect_url
                multiline: False
                foreground_color: 1, 1, 1, 1
                background_color: 0.1, 0.1, 0.1, 1
        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Fyers User'
                color: 1, 1, 1, 1
            TextInput:
                id: fyers_user
                multiline: False
                foreground_color: 1, 1, 1, 1
                background_color: 0.1, 0.1, 0.1, 1
        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Fyers Pin'
                color: 1, 1, 1, 1
            TextInput:
                id: fyers_pin
                multiline: False
                foreground_color: 1, 1, 1, 1
                background_color: 0.1, 0.1, 0.1, 1
        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Fyers TOTP'
                color: 1, 1, 1, 1
            TextInput:
                id: fyers_totp
                multiline: False
                foreground_color: 1, 1, 1, 1
                background_color: 0.1, 0.1, 0.1, 1
        BoxLayout:
            size_hint_y: None
            height: dp(30)
            spacing: dp(10)
            Button:
                text: 'Save'
                on_release: root.save_credentials()
                background_normal: ''
                background_color: 0.95, 0.38, 0.25, 1
                color: 1, 1, 1, 1
            Button:
                text: 'Cancel'
                on_release: root.dismiss()
                background_normal: ''
                background_color: 0.95, 0.38, 0.25, 1
                color: 1, 1, 1, 1
'''

fyers = None
fyers_socket = None
ws_token = None

index_symbol = None
interval_minutes = None
quantity = 0
real_trade = False

ist_timezone = pytz.timezone("Asia/Kolkata")

#Variables
ce_ltp = 0
pe_ltp = 0
index_ltp = 0
buy_sell_checked = False
ce_strike = None
pe_strike = None
ce_symbol = None
pe_symbol = None

target = 80
trailing_sl = 40

fixed_ltp = 0
fixed_index_ltp = 0
prev_ltp = 0
target_inside = 0
target_index_inside = 0
trailing_sl_inside = 0
trailing_index_inside = 0

active_order = False

sl_hit_condition = False
total_loss = 0
total_profit = 0
overall_win = 0
overall_loss = 0
total_points = 0
paper_capital = 0
real_capital = 0

unsubscribe_done = False

active_order_sleep = 1

crop_fetched_candle_data = 532

sequence_length = 0
regression_scalers = None

weight_lr_reg = None
weight_lstm_reg = None
weight_gru_reg = None

index_symbols = {
        'Bankex': 'BSE:BANKEX-INDEX',
        'Finnifty': 'NSE:FINNIFTY-INDEX',
        'Bank Nifty': 'NSE:NIFTYBANK-INDEX',
        'Nifty': 'NSE:NIFTY50-INDEX',
        'Sensex': 'BSE:SENSEX-INDEX'
}

reversed_index_symbols = {
    'BSE:BANKEX-INDEX': 'Bankex',
    'NSE:FINNIFTY-INDEX': 'Finnifty',
    'NSE:NIFTYBANK-INDEX': 'Bank Nifty',
    'NSE:NIFTY50-INDEX': 'Nifty',
    'BSE:SENSEX-INDEX': 'Sensex'
}

timeframe_dict = {
    '1 MIN': 1,
    '2 MIN': 2,
    '3 MIN': 3,
    '5 MIN': 5,
    '15 MIN': 15
}

sound_alert = SoundLoader.load('sounds/alert.mp3')
sound_laughing = SoundLoader.load('sounds/evil-man-laughing.mp3')
sound_button = SoundLoader.load('sounds/button.mp3')
sound_success = SoundLoader.load('sounds/success.mp3')
sound_error = SoundLoader.load('sounds/error.mp3')

def fetch_candle_data(number):
    while True:
        try:
            today = date.today()
            yesterday = today - timedelta(number)

            data = {
                "symbol": index_symbol,
                "resolution": interval_minutes,
                "date_format": "1",
                "range_from": yesterday,
                "range_to": today,
                "cont_flag": "1"
            }

            result = fyers.history(data=data)
            
            if result is not None:
                return result
        except Exception as e:
            print(f"Error fetching Candle Data: {e}")
            time.sleep(active_order_sleep)


def process_df_with_features(df):
    ist = timezone('Asia/Kolkata')

    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(ist).dt.tz_localize(None)

    df.set_index(df['datetime'], inplace=True)

    df.drop('datetime', axis=1, inplace=True)
    df.drop('volume', axis=1, inplace=True)

    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    df['high_low_range'] = df['high'] - df['low']
    df['open_close_range'] = df['open'] - df['close']

    df['EMA'] = ta.ema(df['close'])
    df['SMA'] = ta.sma(df['close'])
    df['WMA'] = ta.wma(df['close'])

    macd = ta.macd(df['close'])
    df = pd.concat([df, macd], axis=1)

    df['RSI'] = ta.rsi(df['close'])

    bollinger_bands = ta.bbands(df['close'])
    df = pd.concat([df, bollinger_bands], axis=1)

    stoch_oscillator = ta.stoch(df['high'], df['low'], df['close'])
    df = pd.concat([df, stoch_oscillator], axis=1)
    
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])
    df['CCI'] = ta.cci(df['high'], df['low'], df['close'])    
    df['ROC'] = ta.roc(df['close'])
    df['MOM'] = ta.mom(df['close'])
    df['WPR'] = ta.willr(df['high'], df['low'], df['close'])
    df['Z_Score'] = ta.zscore(df['close'])

    df['Engulfing'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='engulfing')
    df['Hammer'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='hammer')
    df['Hanging_Man'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='hangingman')
    df['Inverted_Hammer'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='invertedhammer')
    df['Shooting_Star'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='shootingstar')
    df['Morning_Star'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='morningstar')
    df['Evening_Star'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='eveningstar')

    significant_max, significant_min = find_local_extrema(df)
    df['Significant_Max'] = 0.0
    df['Significant_Min'] = 0.0
    
    for idx in significant_max:
        df.at[df.index[idx], 'Significant_Max'] = df.at[df.index[idx], 'high']
    
    for idx in significant_min:
        df.at[df.index[idx], 'Significant_Min'] = df.at[df.index[idx], 'low']

    df = df.round(2)

    df.dropna(inplace=True)

    return df


def label_signals(df):
    df['Signal'] = 0
    df['Entry Price'] = 0.0
    df['Exit Price'] = 0.0

    for i in range(len(df)):
        entry_price = df['close'].iloc[i]

        target = df['Target'].iloc[i]
        stop_loss = df['Stop Loss'].iloc[i]

        buy_target_price = entry_price + target
        buy_sl_price = entry_price - stop_loss

        sell_target_price = entry_price - target
        sell_sl_price = entry_price + stop_loss

        future_data = df.iloc[i + 1:]

        # Check for buy signal
        for j in range(len(future_data)):
            future_high = future_data['high'].iloc[j]
            future_low = future_data['low'].iloc[j]

            if future_high >= buy_target_price:
                df.at[df.index[i], 'Signal'] = 2 # Buy Signal
                df.at[df.index[i], 'Entry Price'] = entry_price
                df.at[df.index[i], 'Exit Price'] = future_high
                break
            elif future_low <= buy_sl_price:
                break

        # Check for sell signal
        for j in range(len(future_data)):
            future_high = future_data['high'].iloc[j]
            future_low = future_data['low'].iloc[j]

            if future_low <= sell_target_price:
                df.at[df.index[i], 'Signal'] = 1 # Sell Signal
                df.at[df.index[i], 'Entry Price'] = entry_price
                df.at[df.index[i], 'Exit Price'] = future_low
                break
            elif future_high >= sell_sl_price:
                break

    return df


def sanitize_filename(filename):
    pattern = re.compile(r'[^a-zA-Z0-9_.-]')
    sanitized_filename = pattern.sub('_', filename)
    return sanitized_filename

def is_file_from_past_week(filepath):
    if not os.path.exists(filepath):
        return False
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    return file_mod_time.date() >= (datetime.today().date() - timedelta(days=7))

# Prepare the data for regression
def preprocess_regression_data(df, sequence_length):    
    # Initialize individual scalers for each feature
    regression_scalers = {}
    for column in df.columns:
        regression_scalers[column] = MinMaxScaler()
        df[column] = regression_scalers[column].fit_transform(df[column].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df.iloc[i-sequence_length:i].values)
        y.append(df.iloc[i, df.columns.get_loc('close')])  # Assuming 'close' is always the target
    
    X, y = np.array(X), np.array(y)

    return X, y, regression_scalers

class LoadingAnimation:
    def __init__(self):
        self.loading_popup = None
        self.loading_label = None
        self.loading_event = None

    def show_loading_animation(self):
        self.loading_popup = ModalView(size_hint=(1, 1), background_color=(0, 0, 0, 0.5))
        content = BoxLayout(orientation='vertical', padding=50)
        self.loading_label = Label(text='Loading...', color=(1, 1, 1, 1))
        content.add_widget(self.loading_label)
        self.loading_popup.add_widget(content)
        self.loading_popup.open()
        self.loading_event = Clock.schedule_interval(self.update_loading_text, 0.5)

    def hide_loading_animation(self):
        if self.loading_event:
            self.loading_event.cancel()
        if self.loading_popup:
            self.loading_popup.dismiss()
            self.loading_popup = None

    def update_loading_text(self, dt):
        if self.loading_label:
            current_text = self.loading_label.text
            if current_text.endswith('...'):
                self.loading_label.text = 'Loading'
            else:
                self.loading_label.text += '.'

class AuthScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_file = 'config.json'
        self.load_credentials()
        self.loading_animation = LoadingAnimation()

    def load_credentials(self):
        try:
            with open(self.config_file, 'r') as f:
                self.credentials = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.credentials = {
                "app_id": "",
                "secret_key": "",
                "redirect_url": "",
                "fyers_user": "",
                "fyers_pin": "",
                "fyers_totp": ""
            }

    def modify_credentials(self):
        popup = CredentialsPopup(credentials=self.credentials, save_callback=self.save_credentials)
        popup.open()

    def save_credentials(self, credentials):
        with open(self.config_file, 'w') as f:
            json.dump(credentials, f, indent=4)

    def start_auth(self):
        self.loading_animation.show_loading_animation()
        
        threading.Thread(target=self.authenticate).start()

    def update_loading_text(self, dt):
        if self.loading_label:
            current_text = self.loading_label.text
            if current_text.endswith('...'):
                self.loading_label.text = 'Loading'
            else:
                self.loading_label.text += '.'

    def authenticate(self):
        global fyers, fyers_socket, ws_token

        try:
            import time
            app_id = self.credentials['app_id']
            secret_key = self.credentials['secret_key']
            redirect_uri = self.credentials['redirect_url']
            fyers_user = self.credentials['fyers_user']
            fyers_pin = self.credentials['fyers_pin']
            fyers_totp = self.credentials['fyers_totp']
            
            session = fyersModel.SessionModel(
                client_id=app_id,
                secret_key=secret_key,
                redirect_uri=redirect_uri,
                response_type='code',
                grant_type='authorization_code'
            )

            def get_encoded_string(string):
                string = str(string)
                base64_bytes = base64.b64encode(string.encode("ascii"))
                return base64_bytes.decode("ascii")

            if session is not None:
                session.generate_authcode()

                url_send_login_otp = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
                res = requests.post(url=url_send_login_otp, json={"fy_id": get_encoded_string(fyers_user), "app_id": "2"}).json()

                if datetime.now().second % 30 > 27:
                    time.sleep(5)

                url_verify_otp = "https://api-t2.fyers.in/vagator/v2/verify_otp"
                res2 = requests.post(url=url_verify_otp, json={"request_key": res["request_key"], "otp": pyotp.TOTP(fyers_totp).now()}).json()

                ses = requests.Session()
                url_verify_otp2 = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
                payload2 = {"request_key": res2["request_key"], "identity_type": "pin", "identifier": get_encoded_string(fyers_pin)}
                res3 = ses.post(url=url_verify_otp2, json=payload2).json()

                ses.headers.update({'authorization': f"Bearer {res3['data']['access_token']}"})

                tokenurl = "https://api-t1.fyers.in/api/v3/token"
                payload3 = {
                    "fyers_id": fyers_user,
                    "app_id": app_id[:-4],
                    "redirect_uri": redirect_uri,
                    "appType": "100",
                    "code_challenge": "",
                    "state": "None",
                    "scope": "",
                    "nonce": "",
                    "response_type": "code",
                    "create_cookie": True
                }

                res3 = ses.post(url=tokenurl, json=payload3).json()

                url = res3['Url']
                parsed = urlparse(url)
                auth_code = parse_qs(parsed.query)['auth_code'][0]

                session.set_token(auth_code)

                auth_response = session.generate_token()
                access_token = auth_response["access_token"]

                fyers = fyersModel.FyersModel(client_id=app_id, token=access_token)

                ws_token = app_id + ":" + access_token
                fyers_socket = data_ws.FyersDataSocket(access_token=ws_token, log_path="")

                pd.DataFrame(fyers.get_profile())

                self.move_to_menu()
            else:
                self.show_error_popup("Session creation failed.")
        except Exception as e:
            self.show_error_popup(str(e))
        finally:
            self.loading_animation.hide_loading_animation()

    @mainthread
    def show_error_popup(self, message):
        content = BoxLayout(orientation='vertical', padding=20, spacing=20)
        content.add_widget(Label(text=message, color=(1, 1, 1, 1)))
        close_button = Button(text='Close', size_hint=(1, None), height=(50), background_normal='', background_color=(0.95, 0.38, 0.25, 1), color=(1, 1, 1, 1))
        content.add_widget(close_button)
        error_popup = Popup(title='Error', content=content, size_hint=(0.5, 0.5), auto_dismiss=False)
        close_button.bind(on_release=error_popup.dismiss)
        error_popup.open()

    @mainthread
    def move_to_menu(self):
        sound_laughing.play()
        self.manager.current = 'menu'

class CredentialsPopup(Popup):
    def __init__(self, credentials, save_callback, **kwargs):
        super().__init__(**kwargs)
        self.credentials = credentials
        self.save_callback = save_callback
        self.ids.app_id.text = self.credentials.get('app_id', '')
        self.ids.secret_key.text = self.credentials.get('secret_key', '')
        self.ids.redirect_url.text = self.credentials.get('redirect_url', '')
        self.ids.fyers_user.text = self.credentials.get('fyers_user', '')
        self.ids.fyers_pin.text = self.credentials.get('fyers_pin', '')
        self.ids.fyers_totp.text = self.credentials.get('fyers_totp', '')

    def save_credentials(self):
        self.credentials['app_id'] = self.ids.app_id.text
        self.credentials['secret_key'] = self.ids.secret_key.text
        self.credentials['redirect_url'] = self.ids.redirect_url.text
        self.credentials['fyers_user'] = self.ids.fyers_user.text
        self.credentials['fyers_pin'] = self.ids.fyers_pin.text
        self.credentials['fyers_totp'] = self.ids.fyers_totp.text
        self.save_callback(self.credentials)
        self.dismiss()

class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super(MenuScreen, self).__init__(**kwargs)
        self.temp_overall_win = 0
        self.temp_overall_loss = 0
        self.temp_paper_capital = 0

        Clock.schedule_once(self.register_buttons)

    def on_enter(self):
        global real_capital
        try:
            real_capital = fyers.funds()

            if real_capital != None:
                real_capital = str(real_capital['fund_limit'][0]['equityAmount'])
            else:
                real_capital = 0

            self.ids.label_real_capital.text = str(real_capital)

            temp_trade_data = load_overall()
            if temp_trade_data:
                self.temp_overall_win = temp_trade_data['overall_win']
                self.temp_overall_loss = temp_trade_data['overall_loss']
                self.temp_paper_capital = temp_trade_data['capital']

            self.ids.input_paper_capital.text = str(self.temp_paper_capital)
        except Exception as e:
            print(f"Error Fetching Funds (Menu Screen): {e}")

    def register_buttons(self, dt):
        self.trade_type_buttons = [self.ids.toggle_real_trade, self.ids.toggle_paper_trade]
        self.index_buttons = [self.ids.toggle_nifty, self.ids.toggle_finnity, self.ids.toggle_bank_nifty, self.ids.toggle_bankex, self.ids.toggle_sensex]
        self.timeframe_buttons = [self.ids.toggle_1min, self.ids.toggle_2min, self.ids.toggle_3min, self.ids.toggle_5min, self.ids.toggle_15min]

    def trade_type_selected(self, button, trade_type):
        global real_trade

        sound_button.play()

        for btn in self.trade_type_buttons:
            if btn != button:
                btn.state = 'normal'
                btn.background_normal = ''
                btn.background_color = (0.1, 0.1, 0.1, 1)

        if button.state == 'down':
            real_trade = True if trade_type == 'Real' else False
            button.background_normal = ''
            button.background_color = (0.95, 0.38, 0.25, 1)
        else:
            real_trade = None
            button.background_normal = ''
            button.background_color = (0.1, 0.1, 0.1, 1)

    def index_selected(self, button, index_name):
        global index_symbol

        sound_button.play()

        for btn in self.index_buttons:
            if btn != button:
                btn.state = 'normal'
                btn.background_normal = ''
                btn.background_color = (0.1, 0.1, 0.1, 1)

        if button.state == 'down':
            index_symbol = index_symbols[index_name]
            button.background_normal = ''
            button.background_color = (0.95, 0.38, 0.25, 1)
        else:
            index_symbol = None
            button.background_normal = ''
            button.background_color = (0.1, 0.1, 0.1, 1)

    def timeframe_selected(self, button, timeframe):
        global interval_minutes

        sound_button.play()

        for btn in self.timeframe_buttons:
            if btn != button:
                btn.state = 'normal'
                btn.background_normal = ''
                btn.background_color = (0.1, 0.1, 0.1, 1)

        if button.state == 'down':
            interval_minutes = timeframe_dict[timeframe]
            button.background_normal = ''
            button.background_color = (0.95, 0.38, 0.25, 1)
        else:
            interval_minutes = None
            button.background_normal = ''
            button.background_color = (0.1, 0.1, 0.1, 1)

    def proceed(self):
        global quantity

        if index_symbol != None and interval_minutes != None and real_trade != None and self.ids.quantity_text_input.text.strip() and self.ids.input_paper_capital.text.strip():            
            quantity = self.ids.quantity_text_input.text

            self.temp_paper_capital = int(self.ids.input_paper_capital.text)

            save_overall(self.temp_overall_win, self.temp_overall_loss, self.temp_paper_capital)
            self.manager.current = 'final'
            

def format_capital(capital):
    # Check if the value is negative
    negative = capital < 0
    # Use the absolute value for formatting
    capital = abs(capital)
    
    if capital >= 1_00_00_00_00_00_00_000:  # 1 Shankh
        formatted_value = f'{capital / 1_00_00_00_00_00_00_000:.2f} Shankh'
    elif capital >= 1_00_00_00_00_00_00_000:  # 1 Padma
        formatted_value = f'{capital / 1_00_00_00_00_00_00_000:.2f} Padma'
    elif capital >= 1_00_00_00_00_00_000:  # 1 Nil
        formatted_value = f'{capital / 1_00_00_00_00_00_000:.2f} Nil'
    elif capital >= 1_00_00_00_00_000:  # 1 Kharab
        formatted_value = f'{capital / 1_00_00_00_00_000:.2f} Kharab'
    elif capital >= 1_00_00_00_000:  # 1 Arab
        formatted_value = f'{capital / 1_00_00_00_000:.2f} Arab'
    elif capital >= 1_00_00_000:  # 1 Crore
        formatted_value = f'{capital / 1_00_00_000:.2f} Cr'
    elif capital >= 1_00_000:  # 1 Lakh
        formatted_value = f'{capital / 1_00_000:.2f} L'
    elif capital >= 1_000:  # 1 Thousand
        formatted_value = f'{capital / 1_000:.2f} K'
    else:
        formatted_value = f'{capital:.2f}'

    # Prepend the negative sign if needed
    return f'-{formatted_value}' if negative else formatted_value

# Function to format trade time
def format_trade_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{int(hours)} Hour{'s' if hours > 1 else ''} {int(minutes)} Min{'s' if minutes > 1 else ''}"
    elif minutes > 0:
        return f"{int(minutes)} Min{'s' if minutes > 1 else ''}"
    else:
        return f"{int(seconds)} Sec{'S' if seconds > 1 else ''}"

class BacktestScreen(Screen):
    def start_backtest(self):
        if self.ids.text_input_backtest_data.text.strip() and self.ids.text_input_backtest_initial_capital.text.strip() and self.ids.text_input_backtest_brokerage.text.strip():
            Clock.schedule_once(lambda dt: self.update_button_disabled_true(self.ids.button_backtest_next))
            Clock.schedule_once(lambda dt: self.update_button_disabled_true(self.ids.button_backtest_start))
            Clock.schedule_once(lambda dt: self.update_button_disabled_true(self.ids.button_backtest_menu))

            threading.Thread(target=self.backtest_logic).start()
            
    def backtest_logic(self):
        backtest_days = int(self.ids.text_input_backtest_data.text)
        if backtest_days > 100:
            backtest_days = 100

        backtest_candles = fetch_candle_data(backtest_days)
        raw_backtest_df = pd.DataFrame(backtest_candles['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        raw_backtest_df = raw_backtest_df.drop_duplicates(subset='datetime', keep='first')

        backtest_capital = int(self.ids.text_input_backtest_initial_capital.text)
        brokerage = int(self.ids.text_input_backtest_brokerage.text)

        sound_alert.play()

        backtest_quantity = int(quantity)
        temp_capital = backtest_capital

        total_backtest_profits = 0
        total_backtest_losses = 0
        total_backtest_brokerage = 0

        backtest_trade_active = False
        backtest_entry_price = 0
        backtest_target_price = 0
        backtest_stop_loss_price = 0

        entry_type = None
        backtest_entry_datetime = None
        backtest_levels = None

        detected_patterns = []

        for i in range(crop_fetched_candle_data, len(raw_backtest_df)):
            if backtest_capital > 2 * temp_capital:
                backtest_quantity *= 2
                temp_capital *= 2

            backtest_df = raw_backtest_df[i-crop_fetched_candle_data:i+1].copy()

            backtest_df = process_df_with_features(backtest_df)

            backtest_df = backtest_df.iloc[:-1]

            # Actual Closing Price
            actual_price = backtest_df['close'].iloc[-1]
            actual_high = backtest_df['high'].iloc[-1]
            actual_low = backtest_df['low'].iloc[-1]

            current_time = backtest_df.index[-1].time()

            backtest_significant_values = np.concatenate([
                    backtest_df['Significant_Max'].loc[backtest_df['Significant_Max'] != 0].values,
                    backtest_df['Significant_Min'].loc[backtest_df['Significant_Min'] != 0].values
                ])
            
            backtest_average_values = average_close_levels(backtest_significant_values)
            backtest_nearest_level = find_nearest_significant(actual_price, backtest_average_values)

            # Actual Closing Price
            actual_price = backtest_df['close'].iloc[-1]
            actual_high = backtest_df['high'].iloc[-1]
            actual_low = backtest_df['low'].iloc[-1]

            current_time = backtest_df.index[-1].time()

            backtest_significant_values = np.concatenate([
                    backtest_df['Significant_Max'].loc[backtest_df['Significant_Max'] != 0].values,
                    backtest_df['Significant_Min'].loc[backtest_df['Significant_Min'] != 0].values
                ])
            
            backtest_average_values = average_close_levels(backtest_significant_values)
            backtest_nearest_level = find_nearest_significant(actual_price, backtest_significant_values)

            if current_time >= dt_time(9, (15 + interval_minutes)) and current_time <= dt_time(15, 0):
                if not backtest_trade_active and entry_type == None:
                    curr_atr = backtest_df['ATR'].iloc[-1]
                    curr_momentum = backtest_df['MOM'].iloc[-1]
                    curr_rsi = backtest_df['RSI'].iloc[-1]

                    if (actual_price >= backtest_nearest_level and curr_rsi <= 50 and 
                        (curr_momentum <= 0 and (
                            backtest_df['Engulfing'].iloc[-1] == 100 or
                            backtest_df['Hammer'].iloc[-1] != 0 or
                            backtest_df['Morning_Star'].iloc[-1] != 0 or
                            backtest_df['Inverted_Hammer'].iloc[-1] != 0
                        ))):

                        if backtest_df['Engulfing'].iloc[-1] == 100:
                            detected_patterns.append('Bullish Engulfing')
                        if backtest_df['Hammer'].iloc[-1] != 0:
                            detected_patterns.append('Hammer')
                        if backtest_df['Morning_Star'].iloc[-1] != 0:
                            detected_patterns.append('Morning Star')
                        if backtest_df['Inverted_Hammer'].iloc[-1] != 0:
                            detected_patterns.append('Inverted Hammer')

                        backtest_entry_price = actual_price

                        backtest_target_price = int(actual_price + curr_atr * 2)
                        backtest_stop_loss_price = int(actual_price - curr_atr)

                        backtest_trade_active = True
                        entry_type = "CE"
                        backtest_entry_datetime = backtest_df.index[-1]
                        backtest_levels = backtest_nearest_level

                    elif (actual_price <= backtest_nearest_level and curr_rsi >= 50 and 
                        (curr_momentum >= 0 and (
                            backtest_df['Engulfing'].iloc[-1] == -100 or
                            backtest_df['Hanging_Man'].iloc[-1] != 0 or
                            backtest_df['Evening_Star'].iloc[-1] != 0 or
                            backtest_df['Shooting_Star'].iloc[-1] != 0
                        ))):
                        
                        if backtest_df['Engulfing'].iloc[-1] == -100:
                            detected_patterns.append('Bearish Engulfing')
                        if backtest_df['Hanging_Man'].iloc[-1] != 0:
                            detected_patterns.append('Hanging Man')
                        if backtest_df['Evening_Star'].iloc[-1] != 0:
                            detected_patterns.append('Evening Star')
                        if backtest_df['Shooting_Star'].iloc[-1] != 0:
                            detected_patterns.append('Shooting Star')

                        backtest_entry_price = actual_price

                        backtest_target_price = int(actual_price - curr_atr * 2)
                        backtest_stop_loss_price = int(actual_price + curr_atr)

                        backtest_trade_active = True
                        entry_type = "PE"
                        backtest_entry_datetime = backtest_df.index[-1]
                        backtest_levels = backtest_nearest_level
                
                else:
                    if entry_type == "CE":
                        if actual_high >= backtest_target_price:
                            points = int((backtest_target_price - backtest_entry_price))
                            profits = int(points * backtest_quantity)
                            backtest_capital += profits
                            total_backtest_profits += 1
                            total_backtest_brokerage += brokerage

                            win_percentage = round(total_backtest_profits / (total_backtest_profits + total_backtest_losses) * 100, 2)
                            loss_percentage = round(total_backtest_losses / (total_backtest_profits + total_backtest_losses) * 100, 2)

                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_capital, format_capital(backtest_capital)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_buysell, entry_type))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_points, points))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_win_pct, f'{total_backtest_profits} / {win_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_profitloss, format_capital(profits)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_loss_pct, f'{total_backtest_losses} / {loss_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_brokerage, format_capital(total_backtest_brokerage)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_quantity, str(backtest_quantity)))

                            backtest_trade_active = False
                            entry_type = None
                            detected_patterns = []

                        elif actual_low <= backtest_stop_loss_price:
                            points = int((backtest_stop_loss_price - backtest_entry_price))
                            profits = int(points * backtest_quantity)
                            backtest_capital += profits
                            total_backtest_losses += 1
                            total_backtest_brokerage += brokerage

                            win_percentage = round(total_backtest_profits / (total_backtest_profits + total_backtest_losses) * 100, 2)
                            loss_percentage = round(total_backtest_losses / (total_backtest_profits + total_backtest_losses) * 100, 2)

                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_capital, format_capital(backtest_capital)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_buysell, entry_type))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_points, points))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_win_pct, f'{total_backtest_profits} / {win_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_profitloss, format_capital(profits)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_loss_pct, f'{total_backtest_losses} / {loss_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_brokerage, format_capital(total_backtest_brokerage)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_quantity, str(backtest_quantity)))

                            backtest_trade_active = False
                            entry_type = None
                            detected_patterns = []
                    
                    elif entry_type == "PE":
                        if actual_low <= backtest_target_price:
                            points = int((backtest_entry_price - backtest_target_price))
                            profits = int(points * backtest_quantity)
                            backtest_capital += profits
                            total_backtest_profits += 1
                            total_backtest_brokerage += brokerage

                            win_percentage = round(total_backtest_profits / (total_backtest_profits + total_backtest_losses) * 100, 2)
                            loss_percentage = round(total_backtest_losses / (total_backtest_profits + total_backtest_losses) * 100, 2)

                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_capital, format_capital(backtest_capital)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_buysell, entry_type))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_points, points))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_win_pct, f'{total_backtest_profits} / {win_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_profitloss, format_capital(profits)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_loss_pct, f'{total_backtest_losses} / {loss_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_brokerage, format_capital(total_backtest_brokerage)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_quantity, str(backtest_quantity)))

                            backtest_trade_active = False
                            entry_type = None
                            detected_patterns = []

                        elif actual_high >= backtest_stop_loss_price:
                            points = int((backtest_entry_price - backtest_stop_loss_price))
                            profits = int(points * backtest_quantity)
                            backtest_capital += profits
                            total_backtest_losses += 1
                            total_backtest_brokerage += brokerage

                            win_percentage = round(total_backtest_profits / (total_backtest_profits + total_backtest_losses) * 100, 2)
                            loss_percentage = round(total_backtest_losses / (total_backtest_profits + total_backtest_losses) * 100, 2)

                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_capital, format_capital(backtest_capital)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_buysell, entry_type))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_points, points))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_win_pct, f'{total_backtest_profits} / {win_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_profitloss, format_capital(profits)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_loss_pct, f'{total_backtest_losses} / {loss_percentage}%'))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_brokerage, format_capital(total_backtest_brokerage)))
                            Clock.schedule_once(lambda dt: self.update_label_data(self.ids.label_backtest_quantity, str(backtest_quantity)))

                            backtest_trade_active = False
                            entry_type = None
                            detected_patterns = []
            
            else:
                backtest_trade_active = False
                entry_type = None
                detected_patterns = []

        sound_alert.play()

        Clock.schedule_once(lambda dt: self.update_button_disabled_false(self.ids.button_backtest_next))
        Clock.schedule_once(lambda dt: self.update_button_disabled_false(self.ids.button_backtest_start))
        Clock.schedule_once(lambda dt: self.update_button_disabled_false(self.ids.button_backtest_menu))

    def update_label_data(self, label, text):
        label.text = str(text)

        if text == 'Buy':
            label.color = (0, 1, 0, 1)
        elif text == 'Sell':
            label.color = (1, 0, 0, 1)
        else:
            label.color = (1, 1, 1, 1)

    def update_button_disabled_true(self, button):
        button.disabled = True

    def update_button_disabled_false(self, button):
        button.disabled = False

def get_sleep_time(interval_minutes, market_start_hour=9, market_start_minute=15):
    now = datetime.now()
    market_start_time = now.replace(hour=market_start_hour, minute=market_start_minute, second=0, microsecond=0)
    
    if now < market_start_time:
        # If current time is before the market starts, set next_run_time to market start time
        next_run_time = market_start_time
    else:
        # Calculate the minutes since the market start time
        minutes_since_market_start = (now - market_start_time).total_seconds() // 60
        # Calculate the number of minutes to the next interval boundary
        minutes_to_next_interval = interval_minutes - (minutes_since_market_start % interval_minutes)
        # Calculate the next run time by adding these minutes to the current time
        next_run_time = (now + timedelta(minutes=minutes_to_next_interval)).replace(second=0, microsecond=0)

    # Calculate the sleep time in seconds
    sleep_time = (next_run_time - now).total_seconds()
    return sleep_time

def fetch_option_chain():
    while True:
        try:
            data = {
                "symbol": index_symbol,
                "strikecount": 2,
                "timestamp": ""
            }
            response = fyers.optionchain(data=data)

            if response is not None:
                return response
        except Exception as e:
            print(f"Error fetching Option Chain: {e}")
            time.sleep(active_order_sleep)

def assign_ce_pe_option_symbols():
    symbol_oc = fetch_option_chain()

    if symbol_oc != None:
        # Convert the response data into a DataFrame
        oc_df = pd.DataFrame(symbol_oc['data']['optionsChain'])

        # Find the first 'CE' symbol from the top
        first_ce_symbol = None
        for index, row in oc_df.iterrows():
            if row['option_type'] == 'CE':
                first_ce_symbol = row['symbol']
                first_ce_strike = row['strike_price']
                break

        # Find the first 'PE' symbol from the bottom
        first_pe_symbol = None
        for index, row in oc_df[::-1].iterrows():  # Iterate in reverse
            if row['option_type'] == 'PE':
                first_pe_symbol = row['symbol']
                first_pe_strike = row['strike_price']
                break

        return first_ce_symbol, first_pe_symbol, first_ce_strike, first_pe_strike

def onmessage_ce(ce_message):
    global ce_ltp, index_ltp, unsubscribe_done
    try:
        if ce_message['symbol'] == ce_symbol:
            if "ltp" in ce_message:
                ce_ltp = ce_message["ltp"]
                ce_ltp = float(ce_ltp)

        elif ce_message['symbol'] == index_symbol:
            if "ltp" in ce_message:
                index_ltp = ce_message["ltp"]
                index_ltp = float(index_ltp)

        if sl_hit_condition and not unsubscribe_done:
            data_type = "SymbolUpdate"
            symbols_to_unsubscribe = [ce_symbol, index_symbol]
            fyers_socket.unsubscribe(symbols=symbols_to_unsubscribe, data_type=data_type)

            unsubscribe_done = True  # Set the flag to True after unsubscribing

            print(f"Unsubscribed {ce_symbol} & {index_symbol}")

    except Exception as e:
        print(f"Error (onMessageCE): {e}")

def onerror_ce(message):
    print("CE LTP Error:", message)


def onclose_ce(message):
    print("CE Connection closed:", message)


def onopen_ce():

    # Specify the data type and symbols you want to subscribe to
    data_type = "SymbolUpdate"

    # Subscribe to the specified symbols and data type
    symbols = [ce_symbol, index_symbol]
    fyers_socket.subscribe(symbols=symbols, data_type=data_type)

    # Keep the socket running to receive real-time data
    fyers_socket.keep_running()
    
# Function to fetch and return the Call Option's Last Traded Price (LTP), strike, and symbol.
def ce_buy_sell_ltp():
    global buy_sell_checked, ce_symbol, ce_strike
    try:
        if not buy_sell_checked:
            buy_sell_checked = True

            print("Fetching CE Strike Price LTP")

            ce_symbol, pe_symbol, ce_strike, pe_strike = assign_ce_pe_option_symbols()

            if ce_symbol is not None and ce_strike is not None:
                # Create a FyersDataSocket instance with the provided parameters
                ce_socket_fyers = data_ws.FyersDataSocket(
                    access_token=ws_token,       # Access token in the format "appid:accesstoken"
                    log_path="",                     # Path to save logs. Leave empty to auto-create logs in the current directory.
                    litemode=True,                  # Lite mode disabled. Set to True if you want a lite response.
                    write_to_file=False,              # Save response in a log file instead of printing it.
                    reconnect=True,                  # Enable auto-reconnection to WebSocket on disconnection.
                    on_connect=onopen_ce,               # Callback function to subscribe to data upon connection.
                    on_close=onclose_ce,                # Callback function to handle WebSocket connection close events.
                    on_error=onerror_ce,                # Callback function to handle WebSocket errors.
                    on_message=onmessage_ce             # Callback function to handle incoming messages from the WebSocket.
                )

                # Establish a connection to the Fyers WebSocket
                ce_socket_fyers.connect()
    
    except Exception as e:
        print(f"Error fetching CE Buy/Sell LTP: {e}")

def onmessage_pe(pe_message):
    global pe_ltp, index_ltp, unsubscribe_done
    try:
        if pe_message['symbol'] == pe_symbol:
            if "ltp" in pe_message:
                pe_ltp = pe_message["ltp"]
                pe_ltp = float(pe_ltp)

        elif pe_message['symbol'] == index_symbol:
            if "ltp" in pe_message:
                index_ltp = pe_message["ltp"]
                index_ltp = float(index_ltp)

        if sl_hit_condition and not unsubscribe_done:
            data_type = "SymbolUpdate"
            symbols_to_unsubscribe = [pe_symbol, index_symbol]
            fyers_socket.unsubscribe(symbols=symbols_to_unsubscribe, data_type=data_type)

            unsubscribe_done = True  # Set the flag to True after unsubscribing

            print(f"Unsubscribed {pe_symbol} & {index_symbol}")

    except Exception as e:
        print(f"Error (onMessagePE): {e}")

def onerror_pe(message):
    print("PE LTP Error:", message)


def onclose_pe(message):
    print("PE Connection closed:", message)


def onopen_pe():    
    # Specify the data type and symbols you want to subscribe to
    data_type = "SymbolUpdate"

    # Subscribe to the specified symbols and data type
    symbols = [pe_symbol, index_symbol]
    fyers_socket.subscribe(symbols=symbols, data_type=data_type)

    # Keep the socket running to receive real-time data
    fyers_socket.keep_running()
    
# Function to fetch and return the Call Option's Last Traded Price (LTP), strike, and symbol.
def pe_buy_sell_ltp():
    global buy_sell_checked, pe_symbol, pe_strike
    try:
        if not buy_sell_checked:
            buy_sell_checked = True

            print("Fetching PE Strike Price LTP")

            ce_symbol, pe_symbol, ce_strike, pe_strike = assign_ce_pe_option_symbols()

            if pe_symbol is not None and pe_strike is not None:
                # Create a FyersDataSocket instance with the provided parameters
                pe_socket_fyers = data_ws.FyersDataSocket(
                    access_token=ws_token,       # Access token in the format "appid:accesstoken"
                    log_path="",                     # Path to save logs. Leave empty to auto-create logs in the current directory.
                    litemode=True,                  # Lite mode disabled. Set to True if you want a lite response.
                    write_to_file=False,              # Save response in a log file instead of printing it.
                    reconnect=True,                  # Enable auto-reconnection to WebSocket on disconnection.
                    on_connect=onopen_pe,               # Callback function to subscribe to data upon connection.
                    on_close=onclose_pe,                # Callback function to handle WebSocket connection close events.
                    on_error=onerror_pe,                # Callback function to handle WebSocket errors.
                    on_message=onmessage_pe             # Callback function to handle incoming messages from the WebSocket.
                )

                # Establish a connection to the Fyers WebSocket
                pe_socket_fyers.connect()
    
    except Exception as e:
        print(f"Error fetching CE Buy/Sell LTP: {e}")

def place_order(symbol):
    if real_trade:
        try:            
            market_order_data = {
                "symbol": symbol,
                "qty": int(quantity),
                "type": 2,  # Market Order
                "side": 1,
                "productType": "INTRADAY",
                "limitPrice": 0,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder":False
            }
        
            market_order_entry = fyers.place_order(data=market_order_data)
            
            if "id" in market_order_entry:
                market_order_id = market_order_entry["id"]
                market_order_message = market_order_entry["message"]
                print(f"{market_order_message}")
        
        except Exception as e:
            print(f"Error placing orders: {str(e)}")

def trail_order(symbol, stoploss):
    if real_trade:
        while True:
            try:
                stoploss = int(stoploss)
                pending_order = fyers.orderbook()
                
                matching_orders = [order for order in pending_order["orderBook"] if order["status"] == 6]
                
                modified_orders = 0
                
                for order in matching_orders:
                    if order['symbol'] == symbol:
                        pending_order_id = order['id']
                        pending_order_side = order['side']
                        pending_order_side = int(pending_order_side)

                        if pending_order_side != 1:
                            data = {
                                "id": pending_order_id,
                                "type": 4,
                                "limitPrice": stoploss - 1, 
                                "stopPrice": stoploss
                            }
                            
                            modify = fyers.modify_order(data=data)
                            trail_message = modify["message"]
                            print(f"{trail_message}")
                            
                            if trail_message == "Successfully modified order":
                                modified_orders += 1
                
                # Check if all matching orders are successfully modified
                if modified_orders == len(matching_orders):
                    break

                time.sleep(active_order_sleep)
                
            except Exception as e:
                print("Error modifying order:" + str(e))

def exit_active_order(symbol):
    if real_trade:
        try:
            data = {
                "id":f"{symbol}-INTRADAY"
            }

            exit_response = fyers.exit_positions(data=data)

            if ["message"] in exit_response:
                print(exit_response["message"])

        except Exception as e:
            print(f"Error exiting Order: {e}")

def reset_flags():
    global active_order, buy_sell_checked
    
    active_order = False
    buy_sell_checked = False

# Function to save profits and losses
def save_overall(overall_win, overall_loss, paper_capital):
    trade_type = {
        "overall_win": overall_win, 
        "overall_loss": overall_loss,
        "capital": paper_capital
    }
    
    with open("trade_data.json", "w") as file:
        json.dump(trade_type, file)


# Function to load wins and losses
def load_overall():
    try:
        with open('trade_data.json') as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    
def update_labels_text(label, text):
    label.text = str(text)

def update_ltp_chart_layout(final_chart, fig, ax):
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    canvas = FigureCanvasKivyAgg(fig)
    canvas.size_hint_y = None
    canvas.height = dp(700)

    final_chart.clear_widgets()
    final_chart.add_widget(canvas)

def plot_ce_ltp_chart(ax, ce_ltp_array, ce_ltp, fixed_ltp, target_inside, trailing_sl_inside, title):
    ax.clear()

    ax.set_facecolor('black')
    ax.figure.patch.set_facecolor('black')

    # Plot LTP data with labels
    ax.plot(ce_ltp_array, label=f"LTP: {ce_ltp}", color='white')
    ax.axhline(y=fixed_ltp, color='blue', linestyle='--', label=f'Entry LTP: {fixed_ltp}')
    ax.axhline(y=target_inside, color='green', linestyle='-', label=f'Target: {target_inside}')
    ax.axhline(y=trailing_sl_inside, color='red', linestyle='-', label=f'SL: {trailing_sl_inside}')

    # Customize the legend
    legend = ax.legend()
    frame = legend.get_frame()
    frame.set_facecolor((0.1, 0.1, 0.1, 1))  # Dark grey background for the legend
    frame.set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')

    # Set labels and title
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("LTP", color='white')
    ax.set_title(title, color='white')

    # Customize tick colors
    ax.tick_params(colors='white')

    # Remove grid lines
    ax.grid(False)

    # Display the plot
    ax.figure.canvas.draw()

def handle_active_ce_order(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital):
    def ce_order_loop():
        global prev_ltp, target_inside, target_index_inside, trailing_sl_inside, trailing_index_inside, total_profit, total_loss, overall_win, overall_loss, ce_ltp, index_ltp, ce_strike, ce_symbol, sl_hit_condition, total_points, fixed_ltp, fixed_index_ltp, paper_capital, target, trailing_sl, real_capital

        profit_money = 0
        profit_percentage = 0
        loss_percentage = 0

        ce_ltp_array = []
        index_ltp_array = []

        target_hit_once = False

        fig, ax = plt.subplots()
        fig_index, ax_index = plt.subplots()

        Clock.schedule_once(lambda dt: update_ltp_chart_layout(final_chart, fig, ax))
        Clock.schedule_once(lambda dt: update_ltp_chart_layout(final_index_chart, fig_index, ax_index))

        while True:
            try:
                if ce_ltp != 0 and index_ltp != 0:
                    ce_ltp_array.append(ce_ltp)
                    index_ltp_array.append(index_ltp)

                    trade_data = load_overall()
        
                    if trade_data:
                        overall_win = trade_data["overall_win"]
                        overall_loss = trade_data["overall_loss"]
                        paper_capital = trade_data["capital"]

                    total_trades = overall_win + overall_loss      

                    if overall_win > 0:
                        profit_percentage = (overall_win/total_trades) * 100
                        profit_percentage = round(profit_percentage, 2)
                    
                    if overall_loss > 0:
                        loss_percentage = (overall_loss/total_trades) * 100
                        loss_percentage = round(loss_percentage, 2) 
                        

                    if index_ltp <= trailing_index_inside:
                        sound_error.play()

                        exit_active_order(ce_symbol)

                        points = int(ce_ltp) - int(fixed_ltp)
                        index_points = int(index_ltp) - int(fixed_index_ltp)

                        total_points = total_points + points

                        if index_points > 0:
                            total_profit += 1
                            overall_win += 1
                            total_trades = overall_win + overall_loss
                            
                            if overall_win > 0:
                                profit_percentage = (overall_win/total_trades) * 100
                                profit_percentage = round(profit_percentage, 2)
                            
                            if overall_loss > 0:
                                loss_percentage = (overall_loss/total_trades) * 100
                                loss_percentage = round(loss_percentage, 2)

                            
                        elif index_points < 0:
                            total_loss += 1
                            overall_loss += 1
                            total_trades = overall_win + overall_loss
                            
                            if overall_win > 0:
                                profit_percentage = (overall_win/total_trades) * 100
                                profit_percentage = round(profit_percentage, 2)
                            
                            if overall_loss > 0:
                                loss_percentage = (overall_loss/total_trades) * 100
                                loss_percentage = round(loss_percentage, 2)

                        profit_money = points*int(quantity)

                        if not real_trade:
                            paper_capital = (paper_capital + profit_money)
                            save_overall(overall_win, overall_loss, paper_capital)
                        else:
                            real_capital = fyers.funds()
                            if real_capital != None:
                                real_capital = real_capital['fund_limit'][0]['equityAmount']
                            else:
                                real_capital = 0

                        real_capital = int(real_capital)
                        real_capital = real_capital + profit_money

                        save_overall(overall_win, overall_loss, paper_capital)

                        Clock.schedule_once(lambda dt: plot_ce_ltp_chart(ax, ce_ltp_array, ce_ltp, fixed_ltp, target_inside, trailing_sl_inside, f"Order: {ce_strike} CE"))
                        Clock.schedule_once(lambda dt: plot_ce_ltp_chart(ax_index, index_ltp_array, index_ltp, fixed_index_ltp, target_index_inside, trailing_index_inside, "Index Chart"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_overall_profit, f"{overall_win} ({total_profit}) / {profit_percentage}%"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_overall_loss, f"{overall_loss} ({total_loss}) / {loss_percentage}%"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_points_captured, f"Index: {index_points} Premium: {points}"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_profit_loss, f"{profit_money}"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_capital, f"{format_capital(paper_capital) if not real_trade else format_capital(real_capital)}"))

                        sl_hit_condition = True

                        reset_flags()

                        break

                    elif index_ltp >= target_index_inside:
                        sound_success.play()

                        if not target_hit_once:
                            target_hit_once = True
                            target /= 2
                            stop_loss =  trailing_sl / 10

                            target_inside = int(ce_ltp + target)
                            target_index_inside = int(index_ltp + target)

                            trailing_sl_inside = int(ce_ltp - stop_loss)
                            trailing_index_inside = int(index_ltp - stop_loss)

                            trailing_sl /= 2
                            prev_ltp = trailing_index_inside

                        else:
                            target_inside = int(ce_ltp + target)
                            target_index_inside = int(index_ltp + target)

                    else:
                        if (index_ltp - prev_ltp) >= trailing_sl and target_hit_once:
                            sound_success.play()

                            prev_ltp = index_ltp

                            trailing_sl_inside = int(ce_ltp - trailing_sl)
                            trailing_index_inside = int(index_ltp - trailing_sl)
                    
                    Clock.schedule_once(lambda dt: plot_ce_ltp_chart(ax, ce_ltp_array, ce_ltp, fixed_ltp, target_inside, trailing_sl_inside, f"Order: {ce_strike} CE"))
                    Clock.schedule_once(lambda dt: plot_ce_ltp_chart(ax_index, index_ltp_array, index_ltp, fixed_index_ltp, target_index_inside, trailing_index_inside, "Index Chart"))
                    Clock.schedule_once(lambda dt: update_labels_text(label_overall_profit, f"{overall_win} ({total_profit}) / {profit_percentage}%"))
                    Clock.schedule_once(lambda dt: update_labels_text(label_overall_loss, f"{overall_loss} ({total_loss}) / {loss_percentage}%"))

                time.sleep(active_order_sleep)
            
            except Exception as e:
                print(f"CE Active Order Error: {e}")

    threading.Thread(target=ce_order_loop).start()

def plot_pe_ltp_chart(ax, pe_ltp_array, pe_ltp, fixed_ltp, target_inside, trailing_sl_inside, title):
    ax.clear()

    ax.set_facecolor('black')
    ax.figure.patch.set_facecolor('black')

    # Plot LTP data with labels
    ax.plot(pe_ltp_array, label=f"LTP: {pe_ltp}", color='white')
    ax.axhline(y=fixed_ltp, color='blue', linestyle='--', label=f'Entry LTP: {fixed_ltp}')
    ax.axhline(y=target_inside, color='green', linestyle='-', label=f'Target: {target_inside}')
    ax.axhline(y=trailing_sl_inside, color='red', linestyle='-', label=f'SL: {trailing_sl_inside}')

    # Customize the legend
    legend = ax.legend()
    frame = legend.get_frame()
    frame.set_facecolor((0.1, 0.1, 0.1, 1))  # Dark grey background for the legend
    frame.set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')

    # Set labels and title
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("LTP", color='white')
    ax.set_title(title, color='white')

    # Customize tick colors
    ax.tick_params(colors='white')

    # Remove grid lines
    ax.grid(False)

    # Display the plot
    ax.figure.canvas.draw()

def handle_active_pe_order(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital):
    def pe_order_loop():
        global prev_ltp, target_inside, target_index_inside, trailing_sl_inside, trailing_index_inside, total_profit, total_loss, overall_win, overall_loss, pe_ltp, pe_strike, pe_symbol, sl_hit_condition, total_points, fixed_ltp, target, trailing_sl, real_capital

        profit_money = 0
        profit_percentage = 0
        loss_percentage = 0

        pe_ltp_array = []
        index_ltp_array = []

        target_hit_once = False

        fig, ax = plt.subplots()
        fig_index, ax_index = plt.subplots()

        Clock.schedule_once(lambda dt: update_ltp_chart_layout(final_chart, fig, ax))
        Clock.schedule_once(lambda dt: update_ltp_chart_layout(final_index_chart, fig_index, ax_index))

        while True:
            try:
                if pe_ltp != 0 and index_ltp != 0:
                    pe_ltp_array.append(pe_ltp)
                    index_ltp_array.append(index_ltp)                    

                    trade_data = load_overall()
        
                    if trade_data:
                        overall_win = trade_data["overall_win"]
                        overall_loss = trade_data["overall_loss"]
                        paper_capital = trade_data["capital"]

                    total_trades = overall_win + overall_loss      

                    if overall_win > 0:
                        profit_percentage = (overall_win/total_trades) * 100
                        profit_percentage = round(profit_percentage, 2)
                    
                    if overall_loss > 0:
                        loss_percentage = (overall_loss/total_trades) * 100
                        loss_percentage = round(loss_percentage, 2)   

                    if index_ltp >= trailing_index_inside:
                        sound_error.play()

                        exit_active_order(pe_symbol)

                        points = int(pe_ltp) - int(fixed_ltp)
                        index_points = int(fixed_index_ltp) - int(index_ltp)

                        total_points = total_points + points

                        if index_points > 0:
                            total_profit += 1
                            overall_win += 1
                            total_trades = overall_win + overall_loss
                            
                            if overall_win > 0:
                                profit_percentage = (overall_win/total_trades) * 100
                                profit_percentage = round(profit_percentage, 2)
                            
                            if overall_loss > 0:
                                loss_percentage = (overall_loss/total_trades) * 100
                                loss_percentage = round(loss_percentage, 2)

                            
                        elif index_points < 0:
                            total_loss += 1
                            overall_loss += 1
                            total_trades = overall_win + overall_loss
                            
                            if overall_win > 0:
                                profit_percentage = (overall_win/total_trades) * 100
                                profit_percentage = round(profit_percentage, 2)
                            
                            if overall_loss > 0:
                                loss_percentage = (overall_loss/total_trades) * 100
                                loss_percentage = round(loss_percentage, 2)

                        profit_money = points*int(quantity)

                        if not real_trade:
                            paper_capital = (paper_capital + profit_money)
                            save_overall(overall_win, overall_loss, paper_capital)
                        else:
                            real_capital = fyers.funds()
                            if real_capital != None:
                                real_capital = real_capital['fund_limit'][0]['equityAmount']
                            else:
                                real_capital = 0

                        real_capital = int(real_capital)
                        real_capital = real_capital + profit_money

                        save_overall(overall_win, overall_loss, paper_capital)

                        Clock.schedule_once(lambda dt: plot_pe_ltp_chart(ax, pe_ltp_array, pe_ltp, fixed_ltp, target_inside, trailing_sl_inside, f"Order: {pe_strike} PE"))
                        Clock.schedule_once(lambda dt: plot_pe_ltp_chart(ax_index, index_ltp_array, index_ltp, fixed_index_ltp, target_index_inside, trailing_index_inside, "Index Price"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_overall_profit, f"{overall_win} ({total_profit}) / {profit_percentage}%"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_overall_loss, f"{overall_loss} ({total_loss}) / {loss_percentage}%"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_points_captured, f"Index: {index_points} Premium: {points}"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_profit_loss, f"{profit_money}"))
                        Clock.schedule_once(lambda dt: update_labels_text(label_capital, f"{format_capital(paper_capital) if not real_trade else format_capital(real_capital)}"))

                        sl_hit_condition = True

                        reset_flags()

                        break

                    elif index_ltp <= target_index_inside:
                        sound_success.play()

                        if not target_hit_once:
                            target_hit_once = True
                            target /= 2
                            stop_loss =  trailing_sl / 10

                            target_inside = int(pe_ltp + target)
                            target_index_inside = int(index_ltp - target)

                            trailing_sl_inside = int(pe_ltp - stop_loss)
                            trailing_index_inside = int(index_ltp + stop_loss)

                            trailing_sl /= 2
                            prev_ltp = trailing_index_inside
                        
                        else:
                            target_inside = int(pe_ltp + target)
                            target_index_inside = int(index_ltp - target)

                    else:
                        if (prev_ltp - index_ltp) >= trailing_sl and target_hit_once:
                            sound_success.play()

                            prev_ltp = index_ltp

                            trailing_sl_inside = int(pe_ltp - trailing_sl)
                            trailing_index_inside = int(index_ltp + trailing_sl)

                    Clock.schedule_once(lambda dt: plot_pe_ltp_chart(ax, pe_ltp_array, pe_ltp, fixed_ltp, target_inside, trailing_sl_inside, f"Order: {pe_strike} PE"))
                    Clock.schedule_once(lambda dt: plot_pe_ltp_chart(ax_index, index_ltp_array, index_ltp, fixed_index_ltp, target_index_inside, trailing_index_inside, "Index Price"))
                    Clock.schedule_once(lambda dt: update_labels_text(label_overall_profit, f"{overall_win} ({total_profit}) / {profit_percentage}%"))
                    Clock.schedule_once(lambda dt: update_labels_text(label_overall_loss, f"{overall_loss} ({total_loss}) / {loss_percentage}%"))

                time.sleep(active_order_sleep)
            
            except Exception as e:
                print(f"PE Active Order Error: {e}")

    threading.Thread(target=pe_order_loop).start()

def ce_entry(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital):
    threading.Thread(target=ce_buy_sell_ltp).start()

    def ce_entry_thread():
        global fixed_ltp, fixed_index_ltp, target_inside, target_index_inside, trailing_sl_inside, trailing_index_inside, active_order, prev_ltp

        sound_laughing.play()

        while True:
            if ce_ltp != 0 and index_ltp != 0:
                prev_ltp = index_ltp
                temp_index_ltp = prev_ltp

                temp_ltp = ce_ltp

                target_inside = temp_ltp + target
                target_inside = int(target_inside)

                trailing_sl_inside = temp_ltp - trailing_sl
                trailing_sl_inside = int(trailing_sl_inside)

                target_index_inside = temp_index_ltp + target
                target_index_inside = int(target_index_inside)

                trailing_index_inside = temp_index_ltp - trailing_sl
                trailing_index_inside = int(trailing_index_inside)

                if target_inside <= 0:
                    target_inside = 0
                if trailing_sl_inside <= 0:
                    trailing_sl_inside = 0

                place_order(ce_symbol)

                fixed_ltp = temp_ltp
                fixed_index_ltp = temp_index_ltp

                active_order = True

                handle_active_ce_order(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital)
                
                break
            
            else:
                time.sleep(active_order_sleep)


    threading.Thread(target=ce_entry_thread).start()

def pe_entry(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital):
    threading.Thread(target=pe_buy_sell_ltp).start()

    def pe_entry_thread():
        global fixed_ltp, fixed_index_ltp, target_inside, target_index_inside, trailing_sl_inside, trailing_index_inside, active_order, prev_ltp

        sound_laughing.play()

        while True:
            if pe_ltp != 0 and index_ltp !=0:
                prev_ltp = index_ltp
                temp_index_ltp = prev_ltp

                temp_ltp = pe_ltp

                target_inside = temp_ltp + target
                target_inside = int(target_inside)

                trailing_sl_inside = temp_ltp - trailing_sl
                trailing_sl_inside = int(trailing_sl_inside)

                target_index_inside = temp_index_ltp - target
                target_index_inside = int(target_index_inside)

                trailing_index_inside = temp_index_ltp + trailing_sl
                trailing_index_inside = int(trailing_index_inside)

                if target_inside <= 0:
                    target_inside = 0
                if trailing_sl_inside <= 0:
                    trailing_sl_inside = 0

                place_order(pe_symbol)

                fixed_ltp = temp_ltp
                fixed_index_ltp = temp_index_ltp

                active_order = True

                handle_active_pe_order(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital)
                
                break
            
            else:
                time.sleep(active_order_sleep)


    threading.Thread(target=pe_entry_thread).start()

def market_entry_exit_logic(btn_final_manual_ce, btn_final_manual_pe, btn_final_backtest, btn_final_setting, final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital, final_df):  
    global sl_hit_condition, unsubscribe_done, ce_ltp, pe_ltp, index_ltp, fixed_ltp, fixed_index_ltp, prev_ltp, target_inside, target_index_inside, trailing_sl_inside, trailing_index_inside, ce_strike, pe_strike, ce_symbol, pe_symbol

    ce_ltp = 0
    pe_ltp  =0
    index_ltp = 0
    fixed_ltp = 0
    fixed_index_ltp = 0
    prev_ltp = 0
    target_inside = 0
    target_index_inside = 0
    trailing_sl_inside = 0
    trailing_index_inside = 0
    ce_strike = None
    pe_strike = None
    ce_symbol = None
    pe_symbol = None

    final_current_time = final_df.index[-1].time()
    final_current_momentum = final_df['MOM'].iloc[-1]
    final_current_rsi = final_df['RSI'].iloc[-1]
    actual_closing_price = final_df['close'].iloc[-1]

    final_significant_values = np.concatenate([
            final_df['Significant_Max'].loc[final_df['Significant_Max'] != 0].values,
            final_df['Significant_Min'].loc[final_df['Significant_Min'] != 0].values
        ])
    
    final_average_values = average_close_levels(final_significant_values)
    final_nearest_level = find_nearest_significant(actual_closing_price, final_significant_values)

    if final_current_time >= dt_time(9, (15 + interval_minutes)) and final_current_time <= dt_time(15, 0):
        #CE entry
        if (actual_closing_price >= final_nearest_level and final_current_rsi <= 50 and 
            (final_current_momentum <= 0 and (
                final_df['Engulfing'].iloc[-1] == 100 or
                final_df['Hammer'].iloc[-1] != 0 or
                final_df['Morning_Star'].iloc[-1] != 0 or
                final_df['Inverted_Hammer'].iloc[-1] != 0
            ))):
            if not active_order:
                sl_hit_condition = False
                unsubscribe_done = False

                btn_final_backtest.disabled = True
                btn_final_setting.disabled = True
                btn_final_manual_ce.disabled = True
                btn_final_manual_pe.disabled = True

                ce_entry(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital)

        #PE entry
        elif (actual_closing_price <= final_nearest_level and final_current_rsi >= 50 and 
            (final_current_momentum >= 0 and (
                final_df['Engulfing'].iloc[-1] == -100 or
                final_df['Hanging_Man'].iloc[-1] != 0 or
                final_df['Evening_Star'].iloc[-1] != 0 or
                final_df['Shooting_Star'].iloc[-1] != 0
            ))):
            if not active_order:
                sl_hit_condition = False
                unsubscribe_done = False

                btn_final_backtest.disabled = True
                btn_final_setting.disabled = True
                btn_final_manual_ce.disabled = True
                btn_final_manual_pe.disabled = True
                
                pe_entry(final_chart, final_index_chart, label_overall_profit, label_overall_loss, label_points_captured, label_profit_loss, label_capital)


# Function to find local maxima and minima
def find_local_extrema(df):
    order=5
    atr_multiplier=1.5
    min_distance=5
    
    # Find local maxima and minima
    local_max = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
    local_min = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    
    # Calculate the threshold based on ATR
    threshold = df['ATR'] * atr_multiplier
    
    # Filter by significance
    significant_max = []
    significant_min = []
    
    for idx in local_max:
        if idx > order and idx < len(df) - order:
            high = df['high'].iloc[idx]
            if significant_min:
                low = df['low'].iloc[significant_min[-1]]
                if (high - low) > threshold.iloc[idx]:
                    significant_max.append(idx)
            else:
                significant_max.append(idx)
    
    for idx in local_min:
        if idx > order and idx < len(df) - order:
            low = df['low'].iloc[idx]
            if significant_max:
                high = df['high'].iloc[significant_max[-1]]
                if (high - low) > threshold.iloc[idx]:
                    significant_min.append(idx)
            else:
                significant_min.append(idx)
    
    # Ensure minimum distance
    def filter_by_distance(points, min_distance):
        filtered_points = []
        for i in range(len(points)):
            if not filtered_points or (points[i] - filtered_points[-1]) > min_distance:
                filtered_points.append(points[i])
        return filtered_points
    
    significant_max = filter_by_distance(significant_max, min_distance)
    significant_min = filter_by_distance(significant_min, min_distance)
    
    return significant_max, significant_min

def get_trendline(df, point1, point2, kind='high'):
    x = [point1, point2]
    if kind == 'high':
        y = df['high'].values[x]
    else:
        y = df['low'].values[x]
    coeffs = np.polyfit(x, y, 1)
    trendline = np.polyval(coeffs, range(point1, len(df)))
    full_trendline = [np.nan] * point1 + list(trendline)
    return full_trendline

def average_close_levels(significant_values, percentage_threshold=0.001):
    if len(significant_values) == 0:
        return np.array([])

    # Sort the significant values
    sorted_values = np.sort(significant_values)
    
    # Initialize lists to store averaged values
    averaged_values = []
    i = 0
    while i < len(sorted_values):
        current_value = sorted_values[i]
        group = [current_value]
        
        # Find all values within the percentage threshold
        j = i + 1
        while j < len(sorted_values) and abs((sorted_values[j] - current_value) / current_value) <= percentage_threshold:
            group.append(sorted_values[j])
            j += 1
        
        # Calculate the average of the grouped values
        average_value = np.mean(group)
        average_value = round(average_value, 2)
        averaged_values.append(average_value)
        
        # Move to the next group
        i = j

    return np.array(averaged_values)

def find_nearest_significant(value, significant_values):
    significant_values = np.array(significant_values)
    nearest_value = significant_values[np.argmin(np.abs(significant_values - value))]
    return nearest_value


class FinalScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.active = False

        self.loading_animation = LoadingAnimation()

    def on_enter(self, *args):
        global overall_win, overall_loss, paper_capital, real_capital

        profit_percentage = 0
        loss_percentage = 0

        trade_data = load_overall()

        if trade_data:
            overall_win = trade_data["overall_win"]
            overall_loss = trade_data["overall_loss"]
            paper_capital = trade_data["capital"]

        total_trades = overall_win + overall_loss      

        if overall_win > 0:
            profit_percentage = (overall_win/total_trades) * 100
            profit_percentage = round(profit_percentage, 2)
        
        if overall_loss > 0:
            loss_percentage = (overall_loss/total_trades) * 100
            loss_percentage = round(loss_percentage, 2)

        real_capital = fyers.funds()
        if real_capital != None:
            real_capital = real_capital['fund_limit'][0]['equityAmount']
        else:
            real_capital = 0

        self.ids.label_overall_profit.text = str(f"{overall_win} / {profit_percentage}%")
        self.ids.label_overall_loss.text = str(f"{overall_loss} / {loss_percentage}%")
        self.ids.label_capital.text = str(f"{format_capital(paper_capital) if not real_trade else format_capital(real_capital)}")

        self.active = True
        threading.Thread(target=self.main_logic).start()

    def on_leave(self, *args):
        self.active = False

    def final_backtest_navigation(self):
        if not active_order:
            self.manager.current = 'backtest'

    def final_menu_navigation(self):
        if not active_order:
            self.manager.current = 'menu'

    def final_manual_ce_entry(self):
        global sl_hit_condition, unsubscribe_done

        if not active_order:
            sl_hit_condition = False
            unsubscribe_done = False
            
            self.ids.button_backtest_model.disabled = True
            self.ids.button_setting.disabled = True
            self.ids.button_manual_ce_entry.disabled = True
            self.ids.button_manual_pe_entry.disabled = True
            ce_entry(self.ids.final_charts_layout, self.ids.final_index_charts_layout, self.ids.label_overall_profit, self.ids.label_overall_loss, self.ids.label_points_captured, self.ids.label_profit_loss, self.ids.label_capital)
    
    def final_manual_pe_entry(self):
        global sl_hit_condition, unsubscribe_done

        if not active_order:
            sl_hit_condition = False
            unsubscribe_done = False
            
            self.ids.button_backtest_model.disabled = True
            self.ids.button_setting.disabled = True
            self.ids.button_manual_ce_entry.disabled = True
            self.ids.button_manual_pe_entry.disabled = True
            pe_entry(self.ids.final_charts_layout, self.ids.final_index_charts_layout, self.ids.label_overall_profit, self.ids.label_overall_loss, self.ids.label_points_captured, self.ids.label_profit_loss, self.ids.label_capital)

    def main_logic(self):
        global target, trailing_sl

        while self.active:
            if not active_order:
                self.ids.button_backtest_model.disabled = False
                self.ids.button_setting.disabled = False
                self.ids.button_manual_ce_entry.disabled = False
                self.ids.button_manual_pe_entry.disabled = False
                
                Clock.schedule_once(lambda dt: self.loading_animation.show_loading_animation())

                num_candles = 99

                final_candles = fetch_candle_data(10)
                final_df = pd.DataFrame(final_candles['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                final_df = final_df.drop_duplicates(subset='datetime', keep='first')
                final_df = final_df[-(crop_fetched_candle_data + 1):]
                final_df = process_df_with_features(final_df)
                
                final_df = final_df.iloc[:-1]

                target = final_df['ATR'].iloc[-1] * 2

                trailing_sl = final_df['ATR'].iloc[-1]

                # Candle Patterns
                final_detected_pattern = []

                if final_df['Engulfing'].iloc[-1] == 100:
                    final_detected_pattern.append('Bullish Engulfing')
                if final_df['Hammer'].iloc[-1] != 0:
                    final_detected_pattern.append('Hammer')
                if final_df['Morning_Star'].iloc[-1] != 0:
                    final_detected_pattern.append('Morning Star')
                if final_df['Inverted_Hammer'].iloc[-1] != 0:
                    final_detected_pattern.append('Inverted Hammer')

                if final_df['Engulfing'].iloc[-1] == -100:
                    final_detected_pattern.append('Bearish Engulfing')
                if final_df['Hanging_Man'].iloc[-1] != 0:
                    final_detected_pattern.append('Hanging Man')
                if final_df['Evening_Star'].iloc[-1] != 0:
                    final_detected_pattern.append('Evening Star')
                if final_df['Shooting_Star'].iloc[-1] != 0:
                    final_detected_pattern.append('Shooting Star')

                # Identify most recent high and low points
                recent_highs, recent_lows = find_local_extrema(final_df)

                most_recent_high = recent_highs[-1] if len(recent_highs) > 1 else None
                most_recent_low = recent_lows[-1] if len(recent_lows) > 1 else None

                high_trendline = [np.nan] * len(final_df)
                low_trendline = [np.nan] * len(final_df)

                if most_recent_high is not None:
                    previous_high = recent_highs[-2] if len(recent_highs) > 2 else most_recent_high
                    high_trendline = get_trendline(final_df, previous_high, most_recent_high, kind='high')

                if most_recent_low is not None:
                    previous_low = recent_lows[-2] if len(recent_lows) > 2 else most_recent_low
                    low_trendline = get_trendline(final_df, previous_low, most_recent_low, kind='low')

                # Prepare candlestick data for mplfinance
                actual_candles = final_df[-num_candles:].copy()

                # Create a DataFrame for mplfinance
                mpf_df = actual_candles[['open', 'high', 'low', 'close']]

                Clock.schedule_once(lambda dt: self.plot_chart(num_candles, mpf_df, final_df, most_recent_high, most_recent_low, high_trendline, low_trendline))

                Clock.schedule_once(lambda dt: self.loading_animation.hide_loading_animation())

                Clock.schedule_once(lambda dt: self.update_label_final_screen(self.ids.label_candle_pattern, final_detected_pattern))

                #market_entry_exit_logic(self.ids.button_manual_ce_entry, self.ids.button_manual_pe_entry, self.ids.button_backtest_model, self.ids.button_setting, self.ids.final_charts_layout, self.ids.final_index_charts_layout, self.ids.label_overall_profit, self.ids.label_overall_loss, self.ids.label_points_captured, self.ids.label_profit_loss, self.ids.label_capital, final_df)
                
            sleep_time = get_sleep_time(interval_minutes)
            time.sleep(sleep_time)

    def update_label_final_screen(self, label_candle_pattern, final_detected_pattern):
        final_detected_pattern = ', '.join(final_detected_pattern)
        label_candle_pattern.text = str(final_detected_pattern)

    def plot_chart(self, num_candles, mpf_df, final_df, most_recent_high, most_recent_low, high_trendline, low_trendline):

        # Create addplot elements for predicted prices and actual close prices
        ap = [
            mpf.make_addplot(final_df['close'][-num_candles:], color='none', panel=0, secondary_y=False, label=f'Actual Price: {final_df['close'].iloc[-1]}'),
            #mpf.make_addplot(y_pred_ensemble_final_plot, color=(0.95, 0.38, 0.25, 1), panel=0, secondary_y=False, label=f'Predicted Prices ({y_pred_ensemble_final_plot[-1]:.2f})')
        ]

        # Add trendlines to the plot
        if most_recent_high is not None:
            ap.append(mpf.make_addplot(high_trendline[-num_candles:], color='white', linestyle='-', panel=0, secondary_y=False))

        if most_recent_low is not None:
            ap.append(mpf.make_addplot(low_trendline[-num_candles:], color='white', linestyle='-', panel=0, secondary_y=False))

        # Add significant levels to the plot with conditional coloring
        last_closing_price = final_df['close'].iloc[-1]

        final_significant_values = np.concatenate([
                final_df['Significant_Max'][-num_candles:].loc[final_df['Significant_Max'] != 0].values,
                final_df['Significant_Min'][-num_candles:].loc[final_df['Significant_Min'] != 0].values
            ])

        final_average_values = average_close_levels(final_significant_values)

        for level in final_average_values:
            if level != 0:
                color = 'red' if level > last_closing_price else 'green'
                ap.append(mpf.make_addplot([level] * num_candles, color=color, linestyle='-', panel=0, secondary_y=False))

        fig, axlist = mpf.plot(mpf_df, type='candle', style='binancedark', volume=False, addplot=ap,
                                title=f'', ylabel='Price',
                                figsize=(21, 14), returnfig=True)

        for ax in axlist:
            ax.grid(False)

        axlist[0].legend()

        fig.patch.set_facecolor('black')

        canvas = FigureCanvasKivyAgg(fig)
        canvas.size_hint_y = None
        canvas.height = dp(700)

        # Clear any existing widgets in the layout
        self.ids.final_charts_layout.clear_widgets()

        # Add the Matplotlib figure to the Kivy layout
        self.ids.final_charts_layout.add_widget(canvas)

class MyApp(App):
    def build(self):
        Builder.load_string(KV)
        sm = ScreenManager()
        sm.add_widget(AuthScreen(name='auth'))
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(BacktestScreen(name='backtest'))
        sm.add_widget(FinalScreen(name='final'))
        return sm

if __name__ == '__main__':
    MyApp().run()