#pragma once

#include <c2ba/maths/types.hpp>
#include <c2ba/maths/geometry.hpp>
#include <c2ba/maths/numeric.hpp>

struct GLFWwindow;

namespace l2n {

class Camera;

class ViewController {
public:
    ViewController(GLFWwindow* window, float speed = 1.f) :
        m_pWindow(window), m_fSpeed(speed) {
    }

    void setSpeed(float speed) {
        m_fSpeed = speed;
    }

    float getSpeed() const {
        return m_fSpeed;
    }

    void increaseSpeed(float delta) {
        m_fSpeed += delta;
        m_fSpeed = c2ba::max(m_fSpeed, 0.f);
    }

    float getCameraSpeed() const {
        return m_fSpeed;
    }

    bool update(float elapsedTime);

    void setViewMatrix(const c2ba::float4x4& viewMatrix) {
        m_ViewMatrix = viewMatrix;
        m_RcpViewMatrix = c2ba::inverse(viewMatrix);
    }

    const c2ba::float4x4& getViewMatrix() const {
        return m_ViewMatrix;
    }

    const c2ba::float4x4& getRcpViewMatrix() const {
        return m_RcpViewMatrix;
    }

private:
    GLFWwindow* m_pWindow = nullptr;
    float m_fSpeed = 0.f;
    bool m_LeftButtonPressed = false;
    c2ba::double2 m_LastCursorPosition;

    c2ba::float4x4 m_ViewMatrix = c2ba::float4x4(1);
    c2ba::float4x4 m_RcpViewMatrix = c2ba::float4x4(1);
};

}