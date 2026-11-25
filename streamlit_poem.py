# app.py
import os, re, html, base64, mimetypes
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


# STEP 1: Vision LLM
def describe_image(img_bytes, mime):
    b64 = base64.b64encode(img_bytes).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": "Describe this photo in 3 lines: objects, mood, colors."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])
    return model.invoke([msg]).text


#  STEP 2: Tools
@tool
def poem(description: str, lines: int = 10) -> str:
    """Write ONE poem based on a photo description."""
    return model.invoke(
        f"Write a {lines}-line poem based on this photo description:\n{description}\nOnly the poem."
    ).text

@tool
def story(description: str, words: int = 250) -> str:
    """Write ONE short story based on a photo description."""
    return model.invoke(
        f"Write a story of about {words} words based on this photo description:\n{description}\nOnly the story."
    ).text

@tool
def rating_tool(text: str, kind: str) -> str:
    """Rate a poem or story out of 10. kind must be 'poem' or 'story'."""
    return model.invoke(
        f"""Rate this {kind} out of 10 for creativity, emotion and language.
Reply in exactly this format:
Score: X/10
Reason: one short sentence

{kind}:
{text}"""
    ).text

@tool
def save_to_file(filename: str, content: str) -> str:
    """Save text to a .txt file."""
    path = os.path.join(OUT, filename + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved: {path}"


# ---------- UI ----------
st.set_page_config(page_title="Photo to Poem / Story", page_icon="🖼️", layout="centered")

st.markdown("""
<style>
.card{
  position:relative; border-radius:22px; padding:48px 34px; margin:22px 0;
  color:#fff; text-align:center; overflow:hidden;
  box-shadow:0 12px 40px rgba(0,0,0,.35);
  background-size:cover; background-position:center;
}
.card .kind{letter-spacing:6px; font-size:12px; opacity:.8; text-transform:uppercase;}
.card .body{font-family:Georgia,'Times New Roman',serif; font-size:20px; line-height:1.9; margin-top:18px;}
.poem .body{font-style:italic;}
.story .body{text-align:left; font-size:18px;}
.story .body p:first-child::first-letter{
  float:left; font-size:64px; line-height:52px; padding:6px 10px 0 0; color:#ffd166;}
.badge{
  display:inline-block; margin-top:26px; padding:8px 18px; border-radius:999px;
  background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.4);
  backdrop-filter:blur(6px); font-size:15px;}
.reason{opacity:.8; font-size:14px; margin-top:8px; font-style:italic;}
</style>
""", unsafe_allow_html=True)

st.title("🖼️ Photo → Poem / Story")
st.caption("Upload a photo. The AI looks at it, writes, and rates its own work.")

file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

c1, c2 = st.columns(2)
kind = c1.radio("What should I write?", ["Poem", "Story"], horizontal=True)
count = c2.slider("How many?", 1, 3, 1)

if file:
    st.image(file, use_container_width=True)

if st.button("✨ Create", disabled=file is None, type="primary"):
    img_bytes = file.getvalue()
    mime = mimetypes.guess_type(file.name)[0] or "image/jpeg"
    img_b64 = base64.b64encode(img_bytes).decode()

    with st.spinner("Looking at your photo..."):
        description = describe_image(img_bytes, mime)

    results = []
    for i in range(1, count + 1):
        with st.spinner(f"Writing {kind.lower()} {i} of {count}..."):
            tool_fn = poem if kind == "Poem" else story
            text = tool_fn.invoke({"description": description})
            rating = rating_tool.invoke({"text": text, "kind": kind.lower()})
            save_to_file.invoke({
                "filename": f"{kind.lower()}_{i}",
                "content": text + "\n\n" + rating,
            })
        results.append((text, rating))

    st.session_state["data"] = {
        "description": description, "kind": kind,
        "img": f"data:{mime};base64,{img_b64}", "results": results,
    }

# ---------- Show results ----------
data = st.session_state.get("data")
if data:
    st.subheader("What the AI saw")
    st.info(data["description"])

    for i, (text, rating) in enumerate(data["results"], 1):
        m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", rating)
        score = m.group(1) if m else "?"
        r = re.search(r"Reason:\s*(.*)", rating, re.S)
        reason = html.escape(r.group(1).strip()) if r else ""

        if data["kind"] == "Poem":
            body = "<br>".join(html.escape(l) for l in text.strip().split("\n"))
            css = "poem"
        else:
            paras = [p for p in text.strip().split("\n") if p.strip()]
            body = "".join(f"<p>{html.escape(p)}</p>" for p in paras)
            css = "story"

        st.markdown(f"""
<div class="card {css}" style="background-image:
  linear-gradient(rgba(10,10,25,.60), rgba(10,10,25,.85)), url('{data["img"]}');">
  <div class="kind">{data["kind"]} #{i}</div>
  <div class="body">{body}</div>
  <div class="badge">⭐ {score} / 10</div>
  <div class="reason">{reason}</div>
</div>
""", unsafe_allow_html=True)

        st.download_button(
            f"⬇️ Download {data['kind'].lower()} #{i}",
            text, file_name=f"{data['kind'].lower()}_{i}.txt", key=f"dl{i}",
        )