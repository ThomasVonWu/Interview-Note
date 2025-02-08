"""
问题：
    要实现一个状态跳转和注册函数的状态机，包含三种状态（ACC，Autopilot、 Manual），支持状态和跳转以及状态的切换

首先，定义所有状态类都需要实现的接口。这些方法包括:
1) 在进入状态时执行的操作（`on_enter`)
2) 在退出状态时执行的操作（`on_exit`)
3) 以及处理事件的方法（`handle_event`）。

1. **状态接口**：所有状态类都继承自`State`，确保每个状态都有统一的方法接口。
2. **具体状态实现**：每个状态类实现了进入、退出和事件处理逻辑。
3. **状态机管理**：
   - 使用字典存储所有可能的状态实例。
   - `set_initial_state` 初始化当前状态，并调用其`on_enter`方法。
   - `handle_event` 根据接收到的事件决定是否切换状态，若需要，则触发状态转换。
4. **注册函数机制**：允许用户注册回调函数，在每次状态变化时执行这些函数。
5. **状态转换**：私有方法`_transition_to`处理退出当前状态、调用回调函数和进入新状态的过程。
"""


### 步骤 1: 定义状态接口
class State:
    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def handle_event(self, event):
        pass


### 步骤 2: 实现具体的状态类
class ACCState(State):
    """
    为每种状态创建具体的实现。例如, ACC、Autopilot 和 Manual 状态。
    """

    def on_enter(self):
        print("Entering ACC mode")

    def on_exit(self):
        print("Exiting ACC mode")

    def handle_event(self, event):
        if event == "autopilot_enabled":
            return "AUTOPILLOT"
        elif event == "manual_override":
            return "MANUAL"
        return None


class AutopilotState(State):
    def on_enter(self):
        print("Entering Autopilot mode")

    def on_exit(self):
        print("Exiting Autopilot mode")

    def handle_event(self, event):
        if event == "acc_enabled":
            return "ACC"
        elif event == "manual_override":
            return "MANUAL"
        return None


class ManualState(State):
    def on_enter(self):
        print("Entering Manual mode")

    def on_exit(self):
        print("Exiting Manual mode")

    def handle_event(self, event):
        if event == "enable_acc":
            return "ACC"
        elif event == "enable_autopilot":
            return "AUTOPILLOT"
        return None


### 步骤 3: 创建状态机类
class StateMachine:
    """
    状态机类负责管理当前状态，并处理状态转换。它还维护一个注册函数列表，在每次状态变化时调用这些函数。
    """

    def __init__(self):
        self.states = {
            "ACC": ACCState(),
            "AUTOPILLOT": AutopilotState(),
            "MANUAL": ManualState(),
        }
        self.current_state = None
        self.callback_functions = []

    def set_initial_state(self, state_name):
        if state_name in self.states:
            self.current_state = self.states[state_name]
            self.current_state.on_enter()
        else:
            raise ValueError("Invalid initial state")

    def register_callback(self, callback_func):
        self.callback_functions.append(callback_func)

    def unregister_callback(self, callback_func):
        try:
            self.callback_functions.remove(callback_func)
        except ValueError:
            pass

    def handle_event(self, event):
        target_state = self.current_state.handle_event(event)
        if target_state:
            self._transition_to(target_state)

    def _transition_to(self, target_state_name):
        if target_state_name in self.states:
            # 调用当前状态的退出方法
            self.current_state.on_exit()

            # 执行所有注册回调函数
            for func in self.callback_functions:
                func()  # 或者传递必要的参数

            # 切换到目标状态并调用进入方法
            self.current_state = self.states[target_state_name]
            self.current_state.on_enter()
        else:
            raise ValueError("Invalid target state")


### 步骤 4: 使用示例
def on_state_change():
    print("State changed callback triggered")


# 创建状态机实例，初始状态为Manual
sm = StateMachine()
sm.set_initial_state("MANUAL")

# 注册回调函数
sm.register_callback(on_state_change)

# 模拟事件处理
sm.handle_event("enable_acc")  # 切换到ACC
sm.handle_event("autopilot_enabled")  # 切换到ACC

# 更多事件处理...
### 关键点解释：
