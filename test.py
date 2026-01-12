import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import display, HTML
from tmp import TreeMemoryPredictor

# Styling for better graphs
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)

# A sample text about AI
TEXT_SOURCE = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans or animals. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating cars, intelligent routing in content delivery networks, and military simulations.
""".strip()

def visualize_text_prediction(text, split_ratio=0.4):
    split_idx = int(len(text) * split_ratio)
    train_text = text[:split_idx]
    test_text = text[split_idx:]
    
    # Initialize Model
    # High decay to keep context relevant, large N_Max for words
    model = TreeMemoryPredictor(n_max=10, decay=0.999, alphabet_autoscale=True)
    
    # 1. Training Phase
    model.fit(list(train_text))
    
    # 2. Testing Phase
    results = [] # list of (char, is_correct)
    correct_count = 0
    
    for char in test_text:
        pred = model.predict()
        is_correct = (pred == char)
        if is_correct:
            correct_count += 1
        print(pred, end='')
        results.append((char, is_correct))
        model.update(char)
        
    # 3. HTML Generation with improved styling
    acc = (correct_count/len(test_text))*100
    
    html_content = f"""
    <div style="border: 1px solid #ccc; border-radius: 8px; overflow: hidden; font-family: 'Consolas', 'Monaco', monospace;">
        <div style="background-color: #333; color: white; padding: 10px 15px; border-bottom: 1px solid #555;">
            <h3 style="margin: 0; font-size: 16px;">🔮 Text Prediction Stream</h3>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.8;">
                Context: {model.n_max} | Decay: {model.decay} | Test Size: {len(test_text)} chars
            </div>
            <div style="font-size: 14px; margin-top: 5px; font-weight: bold; color: {'#4cd137' if acc > 40 else '#e1b12c'};">
                Accuracy: {acc:.2f}%
            </div>
        </div>
        <div style="padding: 20px; background-color: #f9f9f9; line-height: 1.8; font-size: 15px; white-space: pre-wrap; color: #444;">"""
    
    for char, correct in results:
        # Style logic
        if correct:
            color = "#27ae60" # Green
            bg = "#e8f8f5"    # Very light green bg
            border = "1px solid #abebc6"
        else:
            color = "#c0392b" # Red
            bg = "#fdedec"    # Very light red bg
            border = "1px solid #fadbd8"
        
        # Escape HTML characters just in case
        safe_char = char.replace("<", "&lt;").replace(">", "&gt;")
        
        # Span construction
        html_content += f'<span style="color: {color}; background-color: {bg}; padding: 0 1px; border-radius: 2px;">{safe_char}</span>'
        
    html_content += "</div></div>"
    
    display(HTML(html_content))

# Execute
visualize_text_prediction(TEXT_SOURCE, split_ratio=0.3)