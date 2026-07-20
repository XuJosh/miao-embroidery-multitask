"""
苗绣多任务识别系统（GUI 版）
支持加载“原网络”或“剪枝网络”对单张图片进行真伪 / 纹样 / 疵点识别。
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from model_loader import build_model, load_checkpoint, predict, count_parameters


class EmbroideryApp:
    def __init__(self, root):
        self.root = root
        self.root.title('苗绣多任务识别系统（原网络 / 剪枝网络）')
        self.root.geometry('900x650')
        self.root.minsize(800, 550)

        self.model = None
        self.device = None
        self.current_checkpoint = None

        # 默认路径（相对本文件）
        self.defaults = {
            '原始网络': os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                     '..', 'checkpoints_combined_fused_correct',
                                                     'checkpoint_best.pt')),
            '剪枝网络': os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                     'checkpoints', 'checkpoint_best_pruned_amount0.3.pth.gz')),
        }

        self._build_ui()

    def _build_ui(self):
        # ---- 模型选择区 ----
        frm_model = ttk.LabelFrame(self.root, text='模型选择', padding=10)
        frm_model.pack(fill='x', padx=10, pady=5)

        ttk.Label(frm_model, text='网络类型:').grid(row=0, column=0, sticky='w')
        self.cmb_model = ttk.Combobox(frm_model, values=list(self.defaults.keys()),
                                      state='readonly', width=15)
        self.cmb_model.current(1)  # 默认剪枝网络
        self.cmb_model.grid(row=0, column=1, padx=5)
        self.cmb_model.bind('<<ComboboxSelected>>', self._on_model_change)

        ttk.Label(frm_model, text='Checkpoint:').grid(row=0, column=2, sticky='e')
        self.ent_ckpt = ttk.Entry(frm_model)
        self.ent_ckpt.grid(row=0, column=3, sticky='ew', padx=5)
        self.ent_ckpt.insert(0, self.defaults['剪枝网络'])

        ttk.Button(frm_model, text='浏览', command=self._browse_ckpt).grid(row=0, column=4, padx=2)
        self.btn_load = ttk.Button(frm_model, text='加载模型', command=self._load_model_thread)
        self.btn_load.grid(row=0, column=5, padx=5)

        frm_model.columnconfigure(3, weight=1)

        # ---- 图片选择区 ----
        frm_img = ttk.LabelFrame(self.root, text='待识别图片', padding=10)
        frm_img.pack(fill='x', padx=10, pady=5)

        self.ent_img = ttk.Entry(frm_img)
        self.ent_img.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(frm_img, text='浏览图片', command=self._browse_image).pack(side='left', padx=2)
        self.btn_pred = ttk.Button(frm_img, text='开始识别', command=self._predict_thread)
        self.btn_pred.pack(side='left', padx=5)

        # ---- 结果展示区 ----
        frm_result = ttk.Frame(self.root, padding=10)
        frm_result.pack(fill='both', expand=True, padx=10, pady=5)

        # 左侧图片预览
        frm_preview = ttk.LabelFrame(frm_result, text='图片预览', padding=5)
        frm_preview.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.lbl_preview = ttk.Label(frm_preview)
        self.lbl_preview.pack(expand=True)

        # 右侧识别结果
        frm_text = ttk.LabelFrame(frm_result, text='识别结果', padding=10)
        frm_text.grid(row=0, column=1, sticky='nsew')

        self.lbl_auth = ttk.Label(frm_text, text='真伪：未识别', font=('Microsoft YaHei', 14))
        self.lbl_auth.pack(anchor='w', pady=8)
        self.lbl_pattern = ttk.Label(frm_text, text='纹样：未识别', font=('Microsoft YaHei', 14))
        self.lbl_pattern.pack(anchor='w', pady=8)
        self.lbl_defect = ttk.Label(frm_text, text='疵点：未识别', font=('Microsoft YaHei', 14))
        self.lbl_defect.pack(anchor='w', pady=8)

        self.lbl_detail = ttk.Label(frm_text, text='', wraplength=380, justify='left')
        self.lbl_detail.pack(anchor='w', pady=10)

        frm_result.columnconfigure(0, weight=1)
        frm_result.columnconfigure(1, weight=1)
        frm_result.rowconfigure(0, weight=1)

        # ---- 状态栏 ----
        self.status = ttk.Label(self.root, text='就绪：请选择模型并点击“加载模型”', relief='sunken', anchor='w')
        self.status.pack(fill='x', side='bottom', padx=10, pady=(0, 5))

    def _on_model_change(self, event=None):
        key = self.cmb_model.get()
        self.ent_ckpt.delete(0, tk.END)
        self.ent_ckpt.insert(0, self.defaults.get(key, ''))

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title='选择模型 checkpoint',
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
            img.thumbnail((380, 380))
            self.photo = ImageTk.PhotoImage(img)
            self.lbl_preview.config(image=self.photo)
        except Exception as e:
            self.status.config(text=f'图片预览失败: {e}')

    def _set_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        self.btn_load.config(state=state)
        self.btn_pred.config(state=state)

    def _load_model_thread(self):
        ckpt_path = self.ent_ckpt.get().strip()
        if not ckpt_path:
            messagebox.showwarning('提示', '请先选择 checkpoint 路径')
            return
        self._set_busy(True)
        self.status.config(text='正在加载模型，请稍候...')
        threading.Thread(target=self._load_model_worker, args=(ckpt_path,), daemon=True).start()

    def _load_model_worker(self, ckpt_path):
        try:
            model, device = build_model('auto')
            epoch = load_checkpoint(model, ckpt_path, device)
            total, nonzero = count_parameters(model)
            self.model = model
            self.device = device
            self.current_checkpoint = ckpt_path
            info = f'模型已加载 | 设备: {device} | 总参数: {total:,} | 非零参数: {nonzero:,}'
            if epoch is not None:
                info += f' | epoch: {epoch}'
            self.root.after(0, lambda: self.status.config(text=info))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror('加载失败', f'{e}'))
            self.root.after(0, lambda: self.status.config(text='模型加载失败'))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _predict_thread(self):
        img_path = self.ent_img.get().strip()
        if not img_path:
            messagebox.showwarning('提示', '请先选择图片')
            return
        if self.model is None:
            messagebox.showwarning('提示', '请先加载模型')
            return
        self._set_busy(True)
        self.status.config(text='正在推理...')
        threading.Thread(target=self._predict_worker, args=(img_path,), daemon=True).start()

    def _predict_worker(self, img_path):
        try:
            result = predict(self.model, img_path, self.device)
            self.root.after(0, lambda: self._update_result(result))
            self.root.after(0, lambda: self._show_preview(img_path))
            self.root.after(0, lambda: self.status.config(text='识别完成'))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror('识别失败', f'{e}'))
            self.root.after(0, lambda: self.status.config(text='识别失败'))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _update_result(self, result):
        auth = result['auth']
        pat = result['pattern']
        defect = result['defect']

        self.lbl_auth.config(text=f"真伪：{auth['name']}  (置信度 {auth['prob']:.2%})")
        self.lbl_pattern.config(text=f"纹样：{pat['name']}  (置信度 {pat['prob']:.2%})")
        self.lbl_defect.config(text=f"疵点：{defect['name']}  (置信度 {defect['prob']:.2%})")

        detail = (
            f"真伪概率：真品 {auth['probs']['真品 / 手工']:.2%}，伪作 {auth['probs']['伪作 / 机绣']:.2%}\n"
            f"纹样概率：辫绣 {pat['probs']['辫绣']:.2%}，堆绣 {pat['probs']['堆绣']:.2%}，"
            f"马尾绣 {pat['probs']['马尾绣']:.2%}，其他 {pat['probs']['其他']:.2%}，数纱马尾绣 {pat['probs']['数纱马尾绣']:.2%}\n"
            f"疵点概率：无疵点 {defect['probs']['无疵点']:.2%}，有疵点 {defect['probs']['有疵点']:.2%}"
        )
        self.lbl_detail.config(text=detail)


def main():
    root = tk.Tk()
    # 在 Windows 上尝试设置 DPI 感知，使界面更清晰
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = EmbroideryApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
