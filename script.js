const API_URL = "https://xj2025-github-io.onrender.com/api/chat";
const outputDiv = document.getElementById("output");
const inputBox = document.getElementById("input-box");
const sendButton = document.getElementById("send-button");

// 初始化对话上下文（系统提示 + 空历史）
function initMessages() {
    const systemPrompt = {
        role: "system",
        content: "你是玩贴吧抽象的，梗很多，比如丁真，科比等，你说话很抽象，会轻微抽象的骂人，但是不会骂得很脏"
    };
    return [systemPrompt];
}

// 从 sessionStorage 加载或初始化对话
let messages = JSON.parse(sessionStorage.getItem("chatMessages")) || initMessages();

async function sendRequest(userInput) {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ userInput, messages })
        });

        if (!response.ok) throw new Error(await response.text());
        return await response.json();
    } catch (error) {
        outputDiv.innerHTML += `<div class="error">错误：${error.message}</div>`;
        console.error("API Error:", error);
        return null;
    }
}

// 发送消息
sendButton.addEventListener("click", async () => {
    const userInput = inputBox.value.trim();
    if (!userInput) return;

    // 显示用户输入
    outputDiv.innerHTML += `<div><strong>你：</strong>${userInput}</div>`;
    inputBox.value = "";

    // 调用API并更新对话
    const result = await sendRequest(userInput);
    if (result) {
        outputDiv.innerHTML += `<div><strong>d**k$eep：</strong>${result.reply}</div>`;
        messages = result.updatedMessages;  // 更新上下文
        sessionStorage.setItem("chatMessages", JSON.stringify(messages));  // 临时存储
    }

    outputDiv.scrollTop = outputDiv.scrollHeight;
});

// 按Enter发送
inputBox.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendButton.click();
});

// 页面加载时显示历史消息（可选）
window.addEventListener("load", () => {
    if (messages.length > 1) {
        messages.slice(1).forEach(msg => {
            const sender = msg.role === "user" ? "你" : "d**k$eep";
            outputDiv.innerHTML += `<div><strong>${sender}：</strong>${msg.content}</div>`;
        });
    }
});
