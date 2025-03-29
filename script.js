const API_URL = "https://xj2025-flask.fly.dev/api/chat";
const messagesContainer = document.getElementById("messages");
const inputBox = document.getElementById("user-input");  // 修正ID
const sendButton = document.getElementById("send-btn");  // 修正ID

// 初始化对话上下文
function initMessages() {
    const systemPrompt = {
        role: "system",
        content: "你是马栏村红色文化AI助手"
    };
    return [systemPrompt];
}

let messages = JSON.parse(sessionStorage.getItem("chatMessages")) || initMessages();

// 统一的消息添加函数
function addMessage(sender, content, isError = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message message-${sender === "user" ? "user" : "ai"}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${sender === "user" ? "user-avatar" : ""}`;
    avatar.textContent = sender === "user" ? "你" : "AI";

    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender === "user" ? "user" : "ai"}-bubble`;
    if (isError) bubble.style.color = "red";
    bubble.textContent = content;

    messageDiv.append(avatar, bubble);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}




async function sendRequest(userInput) {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            mode: "cors",  // 显式声明CORS模式
            headers: { "Content-Type": "application/json;charset=utf-8" },
            body: JSON.stringify({ userInput, messages })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "请求失败");
        }
        return await response.json();
    } catch (error) {
        showError(`请求失败: ${error.message}`);
        return null;
    }
}

// 发送消息
sendButton.addEventListener("click", async () => {
    const userInput = inputBox.value.trim();
    if (!userInput) return;

    addMessage("user", userInput);
    inputBox.value = "";

    const result = await sendRequest(userInput);
    if (result) {
        addMessage("AI", result.reply);

        if (result.relatedKnowledge?.length > 0) {
            addMessage("AI", "参考知识：\n" + result.relatedKnowledge.join("\n"));
        }

        messages = result.updatedMessages;
        sessionStorage.setItem("chatMessages", JSON.stringify(messages));
    }
});

// 按Enter发送（避免Shift+Enter冲突）
inputBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendButton.click();
    }
});

// 加载历史消息
window.addEventListener("load", () => {
    if (messages.length > 1) {
        messages.slice(1).forEach(msg => {
            const sender = msg.role === "user" ? "user" : "AI";
            addMessage(sender, msg.content);
        });
    }
});
