"""
苗绣多任务识别系统（原网络版）—— CustomTkinter 现代界面
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

from model_loader import build_model, load_checkpoint, predict, count_parameters

ctk.set_appearance_mode('Light')
ctk.set_default_color_theme('blue')

FONT_TITLE = ('Microsoft YaHei', 24, 'bold')
FONT_SUBTITLE = ('Microsoft YaHei', 13)
FONT_CARD_TITLE = ('Microsoft YaHei', 14, 'bold')
FONT_BODY = ('Microsoft YaHei', 12)
FONT_SMALL = ('Microsoft YaHei', 11)
FONT_RESULT = ('Microsoft YaHei', 18, 'bold')

COLOR_SIDEBAR = '#1565C0'
COLOR_CARD = '#ffffff'
COLOR_GREEN = '#2e7d32'
COLOR_RED = '#c62828'


class EmbroideryApp:
    def __init__(self, root):
        self.root = root
        self.root.title('苗绣智能识别系统')
        self.root.geometry('1150x780')
        self.root.minsize(1000, 700)

        self.model = None
        self.device = None
        self.default_ckpt = os.path.normpath(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'embroidery_cls_best.pth')
        )

        self._build_ui()

    def _build_ui(self):
        # 主容器
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color='#f5f7fa')
        self.main_frame.pack(fill='both', expand=True)

        # 左侧边栏
        self.sidebar = ctk.CTkFrame(self.main_frame, width=320, corner_radius=0,
                                    fg_color=COLOR_SIDEBAR)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        self._build_sidebar()

        # 右侧内容区
        self.content = ctk.CTkFrame(self.main_frame, corner_radius=0, fg_color='#f5f7fa')
        self.content.pack(side='left', fill='both', expand=True, padx=25, pady=25)

        self._build_content()

    def _build_sidebar(self):
        # 标题
        ctk.CTkLabel(self.sidebar, text='🧵 苗绣智能识别', font=FONT_TITLE,
                     text_color='white').pack(anchor='w', padx=28, pady=(35, 5))
        ctk.CTkLabel(self.sidebar, text='真伪鉴别 · 纹样分类 · 疵点检测',
                     font=FONT_SUBTITLE, text_color='#bbdefb').pack(anchor='w', padx=28)

        ctk.CTkFrame(self.sidebar, height=2, fg_color='#42a5f5').pack(
            fill='x', padx=28, pady=25)

        # 模型选择
        ctk.CTkLabel(self.sidebar, text='模型 Checkpoint', font=FONT_BODY,
                     text_color='white').pack(anchor='w', padx=28, pady=(10, 5))
        self.ent_ckpt = ctk.CTkEntry(self.sidebar, font=FONT_SMALL, height=38)
        self.ent_ckpt.pack(fill='x', padx=28)
        self.ent_ckpt.insert(0, self.default_ckpt)

        ctk.CTkButton(self.sidebar, text='浏览模型', font=FONT_SMALL, height=34,
                      fg_color='#42a5f5', hover_color='#64b5f6',
                      command=self._browse_ckpt).pack(anchor='w', padx=28, pady=8)

        # 图片选择
        ctk.CTkLabel(self.sidebar, text='待识别图片', font=FONT_BODY,
                     text_color='white').pack(anchor='w', padx=28, pady=(20, 5))
        self.ent_img = ctk.CTkEntry(self.sidebar, font=FONT_SMALL, height=38)
        self.ent_img.pack(fill='x', padx=28)

        ctk.CTkButton(self.sidebar, text='浏览图片', font=FONT_SMALL, height=34,
                      fg_color='#42a5f5', hover_color='#64b5f6',
                      command=self._browse_image).pack(anchor='w', padx=28, pady=8)

        # 加载模型按钮
        self.btn_load = ctk.CTkButton(self.sidebar, text='加载模型', font=FONT_BODY,
                                      height=44, fg_color='white', text_color=COLOR_SIDEBAR,
                                      hover_color='#e3f2fd', command=self._load_model_thread)
        self.btn_load.pack(fill='x', padx=28, pady=(30, 12))

        self.lbl_model_info = ctk.CTkLabel(self.sidebar, text='模型尚未加载',
                                           font=FONT_SMALL, text_color='#bbdefb')
        self.lbl_model_info.pack(anchor='w', padx=28)

        # 底部大按钮：开始识别
        self.btn_pred = ctk.CTkButton(self.sidebar, text='▶  开始识别', font=FONT_TITLE,
                                      height=70, fg_color=COLOR_GREEN, hover_color='#1b5e20',
                                      command=self._predict_thread)
        self.btn_pred.pack(side='bottom', fill='x', padx=28, pady=30)

    def _build_content(self):
        # 上方：图片预览
        preview_card = ctk.CTkFrame(self.content, corner_radius=16, fg_color=COLOR_CARD,
                                    border_width=1, border_color='#e0e0e0')
        preview_card.pack(fill='both', expand=True, pady=(0, 20))

        ctk.CTkLabel(preview_card, text='🖼️ 图片预览', font=FONT_CARD_TITLE,
                     text_color='#37474f').pack(anchor='w', padx=20, pady=(15, 10))

        self.preview_frame = ctk.CTkFrame(preview_card, corner_radius=12,
                                          fg_color='#e3f2fd', border_width=2,
                                          border_color='#90caf9')
        self.preview_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        self.lbl_preview = ctk.CTkLabel(self.preview_frame, text='请选择一张图片',
                                        font=FONT_BODY, text_color='#78909c')
        self.lbl_preview.place(relx=0.5, rely=0.5, anchor='center')

        # 下方：结果区
        result_area = ctk.CTkFrame(self.content, corner_radius=0, fg_color='#f5f7fa')
        result_area.pack(fill='x')

        self.result_cards = {}
        titles = [
            ('auth', '真伪鉴别', '✓', COLOR_GREEN),
            ('pattern', '纹样分类', '✦', '#2196F3'),
            ('defect', '疵点检测', '⚠', COLOR_RED),
        ]
        for i, (key, title, icon, color) in enumerate(titles):
            card = ctk.CTkFrame(result_area, corner_radius=16, fg_color=COLOR_CARD,
                                border_width=1, border_color='#e0e0e0')
            card.grid(row=0, column=i, padx=(0 if i else 0, 18 if i < 2 else 0),
                      sticky='nsew')

            top = ctk.CTkFrame(card, corner_radius=0, fg_color='transparent')
            top.pack(anchor='w', padx=18, pady=(15, 5))

            ctk.CTkLabel(top, text=icon, font=('Segoe UI Emoji', 20),
                         text_color=color).pack(side='left')
            ctk.CTkLabel(top, text=title, font=FONT_BODY,
                         text_color='#78909c').pack(side='left', padx=8)

            lbl_main = ctk.CTkLabel(card, text='未识别', font=FONT_RESULT,
                                    text_color='#263238')
            lbl_main.pack(anchor='w', padx=18, pady=(0, 2))

            lbl_prob = ctk.CTkLabel(card, text='', font=FONT_SMALL,
                                    text_color='#78909c')
            lbl_prob.pack(anchor='w', padx=18)

            self.result_cards[key] = {'main': lbl_main, 'prob': lbl_prob, 'color': color}

        result_area.columnconfigure((0, 1, 2), weight=1)

        # 详细概率
        self.lbl_detail = ctk.CTkLabel(self.content, text='请先加载模型并选择图片。',
                                       font=FONT_SMALL, text_color='#546e7a',
                                       wraplength=900, justify='left')
        self.lbl_detail.pack(anchor='w', pady=(15, 0))

        # 状态栏
        self.status_bar = ctk.CTkFrame(self.root, corner_radius=0, height=36,
                                       fg_color='white')
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(self.status_bar, text='就绪：请点击「加载模型」',
                                         font=FONT_SMALL, text_color='#78909c')
        self.status_label.pack(side='left', padx=20, pady=6)

        self.progress = ctk.CTkProgressBar(self.status_bar, mode='indeterminate',
                                           width=150, height=8)
        self.progress.pack(side='right', padx=20, pady=6)
        self.progress.set(0)

    # ---------- 事件处理 ----------
    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title='选择原网络 checkpoint',
            filetypes=[('PyTorch checkpoint', '*.pt *.pth *.pth.gz'), ('All files', '*.*')]
        )
        if path:
            self.ent_ckpt.delete(0, tk.END)
            self.ent_ckpt.insert(0, path)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title='选择图片',
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp'), ('All files', '*.*')]
        )
        if path:
            self.ent_img.delete(0, tk.END)
            self.ent_img.insert(0, path)
            self._show_preview(path)

    def _show_preview(self, path):
        try:
            img = Image.open(path).convert('RGB')
            # 按预览框大小等比缩放
            img.thumbnail((720, 420))
            self.photo = ImageTk.PhotoImage(img)
            self.lbl_preview.configure(image=self.photo, text='')
            self.lbl_preview.place(relx=0.5, rely=0.5, anchor='center')
        except Exception as e:
            self.status_label.configure(text=f'图片预览失败: {e}')

    def _set_busy(self, busy, text=''):
        if busy:
            self.progress.start()
            self.status_label.configure(text=text, text_color='#1565C0')
        else:
            self.progress.stop()
            self.progress.set(0)
            self.status_label.configure(text=text or '就绪', text_color='#78909c')

    def _load_model_thread(self):
        ckpt_path = self.ent_ckpt.get().strip()
        if not ckpt_path:
            messagebox.showwarning('提示', '请先选择 checkpoint 路径')
            return
        self._set_busy(True, '正在加载模型，请稍候...')
        threading.Thread(target=self._load_model_worker, args=(ckpt_path,), daemon=True).start()

    def _load_model_worker(self, ckpt_path):
        try:
            model, device = build_model('auto')
            epoch = load_checkpoint(model, ckpt_path, device)
            total, _ = count_parameters(model)
            self.model = model
            self.device = device
            info = f'已加载原网络 | {device} | 参数 {total:,}'
            if epoch is not None:
                info += f' | epoch {epoch}'
            self.root.after(0, lambda: self.lbl_model_info.configure(text=info, text_color='white'))
            self.root.after(0, lambda: self._set_busy(False, '模型加载完成'))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror('加载失败', f'{e}'))
            self.root.after(0, lambda: self.lbl_model_info.configure(text='加载失败', text_color='#ffcdd2'))
            self.root.after(0, lambda: self._set_busy(False, '模型加载失败'))

    def _predict_thread(self):
        img_path = self.ent_img.get().strip()
        if not img_path:
            messagebox.showwarning('提示', '请先选择图片')
            return
        if self.model is None:
            messagebox.showwarning('提示', '请先加载模型')
            return
        self._set_busy(True, '正在推理...')
        threading.Thread(target=self._predict_worker, args=(img_path,), daemon=True).start()

    def _predict_worker(self, img_path):
        try:
            result = predict(self.model, img_path, self.device)
            self.root.after(0, lambda: self._update_result(result))
            self.root.after(0, lambda: self._show_preview(img_path))
            self.root.after(0, lambda: self._set_busy(False, '识别完成'))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror('识别失败', f'{e}'))
            self.root.after(0, lambda: self._set_busy(False, '识别失败'))

    def _update_result(self, result):
        auth = result['auth']
        is_real = auth['label'] == 1

        # 真伪结果始终显示
        self.result_cards['auth']['main'].configure(text=auth['name'])
        self.result_cards['auth']['prob'].configure(
            text=f"置信度 {auth['prob']:.2%}")

        if is_real:
            # 真品：显示纹样和疵点
            pat = result['pattern']
            defect = result['defect']
            self.result_cards['pattern']['main'].configure(text=pat['name'])
            self.result_cards['pattern']['prob'].configure(
                text=f"置信度 {pat['prob']:.2%}")
            self.result_cards['defect']['main'].configure(text=defect['name'])
            self.result_cards['defect']['prob'].configure(
                text=f"置信度 {defect['prob']:.2%}")

            detail = (
                f"✅ 鉴定为真品，继续识别纹样与疵点\n"
                f"真伪：真品 {auth['probs']['真品 / 手工']:.2%}，伪作 {auth['probs']['伪作 / 机绣']:.2%}\n"
                f"纹样：辫绣 {pat['probs']['辫绣']:.2%}  堆绣 {pat['probs']['堆绣']:.2%}  "
                f"马尾绣 {pat['probs']['马尾绣']:.2%}  其他 {pat['probs']['其他']:.2%}  "
                f"数纱马尾绣 {pat['probs']['数纱马尾绣']:.2%}\n"
                f"疵点：无疵点 {defect['probs']['无疵点']:.2%}，有疵点 {defect['probs']['有疵点']:.2%}"
            )
        else:
            # 伪品：不显示纹样和疵点
            self.result_cards['pattern']['main'].configure(text='—')
            self.result_cards['pattern']['prob'].configure(text='伪品，跳过纹样识别')
            self.result_cards['defect']['main'].configure(text='—')
            self.result_cards['defect']['prob'].configure(text='伪品，跳过疵点识别')

            detail = (
                f"❌ 鉴定为伪品\n"
                f"真伪：真品 {auth['probs']['真品 / 手工']:.2%}，伪作 {auth['probs']['伪作 / 机绣']:.2%}\n"
                f"系统已终止后续纹样分类与疵点检测。"
            )

        self.lbl_detail.configure(text=detail)


def main():
    root = ctk.CTk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = EmbroideryApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
