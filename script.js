const API_URL = "https://xj2025-github-io.onrender.com/api/chat";
const outputDiv = document.getElementById("output");
const inputBox = document.getElementById("input-box");
const sendButton = document.getElementById("send-button");

// 初始化对话上下文
function initMessages() {
    const systemPrompt = {
        role: "system",
        content: "你是马栏村红色文化AI助手"
    };
    return [systemPrompt];
}

let messages = JSON.parse(sessionStorage.getItem("chatMessages")) || initMessages();

async function sendRequest(userInput) {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                userInput: userInput,
                messages: messages
            })
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

    // 调用API
    const result = await sendRequest(userInput);
    if (result) {
        // 显示AI回复
        outputDiv.innerHTML += `<div><strong>AI：</strong>${result.reply}</div>`;

        // 调试信息：显示检索到的知识（可选）
        if (result.relatedKnowledge && result.relatedKnowledge.length > 0) {
            outputDiv.innerHTML += `<div class="knowledge"><small>参考知识：${result.relatedKnowledge.join("<br>")}</small></div>`;
        }

        // 更新上下文
        messages = result.updatedMessages;
        sessionStorage.setItem("chatMessages", JSON.stringify(messages));
    }

    outputDiv.scrollTop = outputDiv.scrollHeight;
});

// 按Enter发送
inputBox.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendButton.click();
});

// 加载历史消息
window.addEventListener("load", () => {
    if (messages.length > 1) {
        messages.slice(1).forEach(msg => {
            const sender = msg.role === "user" ? "你" : "AI";
            outputDiv.innerHTML += `<div><strong>${sender}：</strong>${msg.content}</div>`;
        });
    }
});