import tkinter as tk
from tkinter import ttk, messagebox

# 1) Pizza 클래스 정의 (이전 예제와 동일)
class Pizza:
    PRESETS = {
        'margherita':    ['mozzarella', 'tomatoes'],
        'prosciutto':    ['mozzarella', 'tomatoes', 'ham'],
        'hawaiian':      ['mozzarella', 'tomatoes', 'ham', 'pineapple'],
        'funghi':        ['mozzarella', 'tomatoes', 'mushrooms'],
        'vegetarian':    ['mozzarella', 'tomatoes', 'bell peppers', 'olives', 'onions'],
        'pepperoni':     ['mozzarella', 'tomatoes', 'pepperoni'],
        'quattro_formaggi': ['mozzarella', 'gorgonzola', 'parmesan', 'goat cheese'],
        'bbq_chicken':   ['mozzarella', 'bbq sauce', 'chicken', 'onions', 'cilantro'],
        'meat_lovers':   ['mozzarella', 'tomatoes', 'pepperoni', 'sausage', 'bacon', 'ham'],
        'seafood':       ['mozzarella', 'tomatoes', 'shrimp', 'squid', 'garlic']
    }

    def __init__(self, ingredients):
        self.ingredients = ingredients

    def __repr__(self):
        return f"Pizza({self.ingredients})"

    @classmethod
    def from_preset(cls, name):
        ingr = cls.PRESETS.get(name)
        if ingr is None:
            raise ValueError(f"Unknown preset: {name}")
        return cls(ingr)

    @classmethod
    def create_many(cls, names):
        return [cls.from_preset(n) for n in names]


# 2) GUI 구성
class PizzaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pizza Preset Creator")
        self.geometry("400x300")

        # 프리셋 리스트박스 (여러 선택 가능)
        self.listbox = tk.Listbox(self, selectmode=tk.MULTIPLE)
        for name in Pizza.PRESETS:
            self.listbox.insert(tk.END, name)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 생성 버튼
        btn = ttk.Button(self, text="Create", command=self.on_create)
        btn.pack(pady=5)

        # 결과 표시 텍스트 박스
        self.output = tk.Text(self, height=6)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def on_create(self):
        selections = [self.listbox.get(i) for i in self.listbox.curselection()]
        self.output.delete('1.0', tk.END)
        if not selections:
            messagebox.showwarning("No Selection", "하나 이상의 프리셋을 선택해주세요.")
            return
        pizzas = Pizza.create_many(selections)
        for p in pizzas:
            self.output.insert(tk.END, repr(p) + "\n")


if __name__ == "__main__":
    app = PizzaApp()
    app.mainloop()
