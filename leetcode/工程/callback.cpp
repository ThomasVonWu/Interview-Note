/*
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
*/

#include <iostream>
#include <unordered_map>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>

// 1. 定义状态枚举
enum class StateType
{
    ACC,
    Autopilot,
    Manual
};

// 2. 定义基类 `State`
class State
{
public:
    virtual ~State() = default;

    // 进入状态时的操作
    virtual void onEnter() = 0;

    // 退出状态时的操作
    virtual void onExit() = 0;

    virtual StateType getType() const = 0;
};

// 3. 实现具体状态类

class ACCState : public State
{
public:
    void onEnter() override
    {
        std::cout << "Entering ACC mode." << std::endl;
    }

    void onExit() override
    {
        std::cout << "Exiting ACC mode." << std::endl;
    }

    StateType getType() const override { return StateType::ACC; }
};

class AutopilotState : public State
{
public:
    void onEnter() override
    {
        std::cout << "Entering Autopilot mode." << std::endl;
    }

    void onExit() override
    {
        std::cout << "Exiting Autopilot mode." << std::endl;
    }

    StateType getType() const override { return StateType::Autopilot; }
};

class ManualState : public State
{
public:
    void onEnter() override
    {
        std::cout << "Entering Manual mode." << std::endl;
    }

    void onExit() override
    {
        std::cout << "Exiting Manual mode." << std::endl;
    }

    StateType getType() const override { return StateType::Manual; }
};

// 4. 实现状态机类
class StateMachine
{
private:
    // 状态实例映射（使用枚举类型作为键）
    std::unordered_map<StateType, std::unique_ptr<State>> stateMap;

    // 当前状态指针（指向基类）
    State *currentState = nullptr;

    // 回调函数列表
    std::vector<std::function<void()>> callbacks;

public:
    StateMachine()
    {
        stateMap[StateType::ACC] = std::make_unique<ACCState>();
        stateMap[StateType::Autopilot] = std::make_unique<AutopilotState>();
        stateMap[StateType::Manual] = std::make_unique<ManualState>();
    }

    // 设置初始状态
    void setInitialState(StateType initialState)
    {
        if (stateMap.find(initialState) != stateMap.end())
        {
            currentState = stateMap[initialState].get();
            currentState->onEnter();
        }
    }

    // 切换状态（根据类型）
    void switchState(StateType newState)
    {
        if (currentState == nullptr || newState == currentState->getType())
        {
            return;
        }

        // 退出当前状态
        currentState->onExit();

        // 更新为新状态并进入
        auto it = stateMap.find(newState);
        if (it != stateMap.end())
        {
            currentState = it->second.get();
            currentState->onEnter();
        }
        else
        {
            // 处理错误情况，例如抛出异常或保持当前状态
            throw std::invalid_argument("Invalid state transition");
        }

        // 执行所有回调函数
        executeCallbacks();
    }

    // 注册回调函数
    void registerCallback(std::function<void()> callback)
    {
        callbacks.push_back(callback);
    }

    // 注销所有回调函数（清空列表）
    void unregisterAllCallbacks()
    {
        callbacks.clear();
    }

private:
    // 执行所有注册的回调函数
    void executeCallbacks() const
    {
        for (const auto &cb : callbacks)
        {
            cb();
        }
    }
};

int main()
{
    // 创建状态机实例
    StateMachine stateMachine;

    // 设置初始状态为 Manual
    stateMachine.setInitialState(StateType::Manual);

    // 注册一个回调函数
    stateMachine.registerCallback([]() {
        std::cout << "Callback executed during state change!" << std::endl;
    });

    // 切换到 ACC 状态
    stateMachine.switchState(StateType::ACC);

    // 再次切换到 Autopilot 状态
    stateMachine.switchState(StateType::Autopilot);

    return 0;
}
