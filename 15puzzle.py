import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import random
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import time
from collections import deque
import threading
import heapq

class FastPDBSolver:
    """軽量版Pattern Database - より実用的な実装"""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.pattern_db = {}
        self.is_initialized = False
        
        # より軽量なパターンを使用
        if grid_size == 4:
            # 7-8パターンではなく、より軽量な5-5-5パターンを使用
            self.patterns = [
                frozenset([1, 2, 3, 4, 5]),      # 上部5タイル
                frozenset([6, 7, 8, 9, 10]),     # 中部5タイル
                frozenset([11, 12, 13, 14, 15])  # 下部5タイル
            ]
        elif grid_size == 3:
            self.patterns = [
                frozenset([1, 2, 3, 4]),
                frozenset([5, 6, 7, 8])
            ]
        else:
            # 5x5は簡単なパターンを使用
            self.patterns = [
                frozenset([1, 2, 3, 4, 5, 6]),
                frozenset([7, 8, 9, 10, 11, 12])
            ]
    
    def initialize_pdb(self, progress_callback=None):
        """軽量版PDBの初期化"""
        if self.is_initialized:
            return
        
        print(f"Initializing lightweight PDB for {self.grid_size}x{self.grid_size}...")
        
        self.pdbs = []
        for i, pattern in enumerate(self.patterns):
            if progress_callback:
                progress_callback(f"Building pattern {i+1}/{len(self.patterns)}...")
            
            # より限定的な深度でPDBを構築（メモリ節約）
            pdb = self._build_limited_pdb(pattern, max_depth=10)
            self.pdbs.append(pdb)
        
        self.is_initialized = True
        print("Lightweight PDB initialized!")
    
    def _build_limited_pdb(self, pattern, max_depth):
        """制限された深度でPDBを構築"""
        db = {}
        queue = deque()
        
        goal_state = self._create_goal_state()
        goal_pattern_state = self._extract_pattern_state(goal_state, pattern)
        
        db[goal_pattern_state] = 0
        queue.append((goal_state, 0))
        
        processed = 0
        while queue and processed < 50000:  # 処理数制限
            current_state, cost = queue.popleft()
            processed += 1
            
            if cost >= max_depth:
                continue
            
            empty_pos = self._find_empty_pos(current_state)
            
            for next_state in self._get_neighbors(current_state, empty_pos):
                pattern_state = self._extract_pattern_state(next_state, pattern)
                
                if pattern_state not in db:
                    db[pattern_state] = cost + 1
                    queue.append((next_state, cost + 1))
        
        return db
    
    def _create_goal_state(self):
        """目標状態を作成"""
        state = []
        num = 1
        for row in range(self.grid_size):
            state_row = []
            for col in range(self.grid_size):
                if row == self.grid_size - 1 and col == self.grid_size - 1:
                    state_row.append(0)
                else:
                    state_row.append(num)
                    num += 1
            state.append(tuple(state_row))
        return tuple(state)
    
    def _find_empty_pos(self, state):
        """空白の位置を見つける"""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state[row][col] == 0:
                    return (row, col)
        return None
    
    def _get_neighbors(self, state, empty_pos):
        """隣接状態を生成"""
        neighbors = []
        empty_row, empty_col = empty_pos
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = empty_row + dr, empty_col + dc
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                new_state = [list(row) for row in state]
                new_state[empty_row][empty_col] = new_state[new_row][new_col]
                new_state[new_row][new_col] = 0
                neighbors.append(tuple(tuple(row) for row in new_state))
        
        return neighbors
    
    def _extract_pattern_state(self, state, pattern):
        """パターンに対応する状態を抽出（簡易版）"""
        pattern_tiles = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                tile = state[row][col]
                if tile in pattern:
                    pattern_tiles.append((tile, row, col))
        return tuple(sorted(pattern_tiles))
    
    def get_heuristic(self, state):
        """PDBを使用してヒューリスティック値を計算"""
        if not self.is_initialized:
            return self._manhattan_distance(state)
        
        # 全パターンの最大値を返す
        max_cost = 0
        for i, pattern in enumerate(self.patterns):
            pattern_state = self._extract_pattern_state(state, pattern)
            cost = self.pdbs[i].get(pattern_state, 0)
            max_cost = max(max_cost, cost)
        
        # マンハッタン距離との最大値も考慮
        manhattan = self._manhattan_distance(state)
        return max(max_cost, manhattan)
    
    def _manhattan_distance(self, state):
        """マンハッタン距離を計算"""
        distance = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state[row][col] != 0:
                    target_num = state[row][col]
                    target_row = (target_num - 1) // self.grid_size
                    target_col = (target_num - 1) % self.grid_size
                    distance += abs(row - target_row) + abs(col - target_col)
        return distance

class FastAStar:
    """A*アルゴリズムの実装 - より堅牢な版"""
    
    def __init__(self, grid_size, pdb_solver=None):
        self.grid_size = grid_size
        self.pdb_solver = pdb_solver
        self.nodes_expanded = 0
        # グリッドサイズに応じて上限を調整
        if grid_size <= 3:
            self.max_nodes = 100000
        elif grid_size == 4:
            self.max_nodes = 1000000
        else:
            self.max_nodes = 500000
    
    def solve(self, initial_state, progress_callback=None):
        """A*アルゴリズムで解を探索"""
        state_tuple = tuple(tuple(row) for row in initial_state)
        
        if self._is_goal(state_tuple):
            return []
        
        # まず解ける状態かチェック
        if not self._is_solvable_state(state_tuple):
            print("Warning: Puzzle state appears to be unsolvable!")
            return None
        
        # 優先度キュー: (f値, g値, 状態, パス)
        heap = [(self._get_heuristic(state_tuple), 0, state_tuple, [])]
        visited = {}  # 状態 -> 最小g値
        
        best_heuristic = float('inf')
        
        while heap and self.nodes_expanded < self.max_nodes:
            f_val, g_val, current_state, path = heapq.heappop(heap)
            
            # 既により良い解で訪問済みならスキップ
            if current_state in visited and visited[current_state] <= g_val:
                continue
            
            visited[current_state] = g_val
            self.nodes_expanded += 1
            
            if self.nodes_expanded % 50000 == 0 and progress_callback:
                h_val = f_val - g_val
                progress_callback(f"探索中: {self.nodes_expanded} nodes, depth: {g_val}, h: {h_val}")
                
                # 進歩がない場合の検出
                if h_val < best_heuristic:
                    best_heuristic = h_val
            
            if self._is_goal(current_state):
                print(f"Solution found! Length: {len(path)}, Nodes expanded: {self.nodes_expanded}")
                return path
            
            # 隣接状態を展開
            empty_pos = self._find_empty_pos(current_state)
            for next_state, move in self._get_neighbors_with_moves(current_state, empty_pos):
                new_g = g_val + 1
                
                # より良い経路で既に訪問済みならスキップ
                if next_state in visited and visited[next_state] <= new_g:
                    continue
                
                new_f = new_g + self._get_heuristic(next_state)
                new_path = path + [move]
                heapq.heappush(heap, (new_f, new_g, next_state, new_path))
        
        print(f"Search completed. Nodes expanded: {self.nodes_expanded}")
        return None  # 解が見つからない
    
    def _is_solvable_state(self, state):
        """状態が解けるかどうかをチェック"""
        # フラットなリストに変換（空白は除く）
        flat_tiles = []
        empty_row = 0
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state[row][col] == 0:
                    empty_row = row
                else:
                    flat_tiles.append(state[row][col])
        
        # 転倒数を計算
        inversions = 0
        for i in range(len(flat_tiles)):
            for j in range(i + 1, len(flat_tiles)):
                if flat_tiles[i] > flat_tiles[j]:
                    inversions += 1
        
        # 解ける条件をチェック
        if self.grid_size % 2 == 1:
            # 奇数サイズ: 転倒数が偶数なら解ける
            return inversions % 2 == 0
        else:
            # 偶数サイズ
            empty_row_from_bottom = self.grid_size - empty_row
            if empty_row_from_bottom % 2 == 0:
                return inversions % 2 == 1
            else:
                return inversions % 2 == 0
    
    def _find_empty_pos(self, state):
        """空白の位置を見つける"""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state[row][col] == 0:
                    return (row, col)
        return None
    
    def _get_neighbors_with_moves(self, state, empty_pos):
        """隣接状態と移動を生成"""
        neighbors = []
        empty_row, empty_col = empty_pos
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = empty_row + dr, empty_col + dc
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                new_state = [list(row) for row in state]
                new_state[empty_row][empty_col] = new_state[new_row][new_col]
                new_state[new_row][new_col] = 0
                new_state_tuple = tuple(tuple(row) for row in new_state)
                neighbors.append((new_state_tuple, (new_row, new_col)))
        
        return neighbors
    
    def _get_heuristic(self, state):
        """ヒューリスティック値を取得"""
        if self.pdb_solver:
            return self.pdb_solver.get_heuristic(state)
        else:
            return self._manhattan_distance(state)
    
    def _manhattan_distance(self, state):
        """マンハッタン距離を計算"""
        distance = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state[row][col] != 0:
                    target_num = state[row][col]
                    target_row = (target_num - 1) // self.grid_size
                    target_col = (target_num - 1) % self.grid_size
                    distance += abs(row - target_row) + abs(col - target_col)
        return distance
    
    def _is_goal(self, state):
        """目標状態かチェック"""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if row == self.grid_size - 1 and col == self.grid_size - 1:
                    if state[row][col] != 0:
                        return False
                else:
                    expected = row * self.grid_size + col + 1
                    if state[row][col] != expected:
                        return False
        return True

class SlidingPuzzle:
    def __init__(self, root):
        self.root = root
        self.root.title("スライディングパズル - 改善版")
        self.root.geometry("600x750")
        self.root.minsize(400, 550)
        
        # パズルの設定
        self.grid_size = 4
        self.tile_size = 100
        self.puzzle_size = self.grid_size * self.tile_size
        
        # 状態管理
        self.original_image = None
        self.puzzle_image = None
        self.tiles = []
        self.empty_pos = (self.grid_size-1, self.grid_size-1)
        self.tile_images = []
        
        # AUTO機能用の変数
        self.is_auto_running = False
        self.auto_moves = []
        self.auto_after_id = None
        self.solver_thread = None
        
        # 改善されたソルバー
        self.pdb_solver = None
        self.astar_solver = None
        
        self.setup_ui()
        self.reset_puzzle()
        
        # ウィンドウリサイズイベント
        self.root.bind('<Configure>', self.on_window_resize)
        
    def setup_ui(self):
        # メインフレーム
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # サイズ選択フレーム
        self.size_frame = tk.Frame(self.main_frame)
        self.size_frame.pack(pady=(0, 5))
        
        tk.Label(self.size_frame, text="パズルサイズ:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
        
        # サイズ選択ボタン
        self.size_var = tk.IntVar(value=4)
        for size in [3, 4, 5]:
            size_name = f"{size}x{size}" + (" (8パズル)" if size == 3 else " (15パズル)" if size == 4 else " (24パズル)")
            rb = tk.Radiobutton(
                self.size_frame,
                text=size_name,
                variable=self.size_var,
                value=size,
                command=self.change_grid_size,
                font=("Arial", 9)
            )
            rb.pack(side=tk.LEFT, padx=5)
        
        # ボタンフレーム
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=(5, 10))
        
        # ボタン
        self.load_button = tk.Button(
            self.button_frame, 
            text="画像を読み込み", 
            command=self.load_image,
            font=("Arial", 10)
        )
        self.load_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.shuffle_button = tk.Button(
            self.button_frame, 
            text="シャッフル", 
            command=self.shuffle_puzzle,
            font=("Arial", 10),
            state=tk.DISABLED
        )
        self.shuffle_button.pack(side=tk.LEFT, padx=5)
        
        self.auto_button = tk.Button(
            self.button_frame, 
            text="AUTO (高速)", 
            command=self.toggle_auto_solve,
            font=("Arial", 10),
            state=tk.DISABLED,
            bg="lightgreen"
        )
        self.auto_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(
            self.button_frame, 
            text="リセット", 
            command=self.reset_puzzle,
            font=("Arial", 10)
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # パズルフレーム（ドラッグアンドドロップ対応）
        self.puzzle_frame = tk.Frame(
            self.main_frame, 
            bg="lightgray", 
            relief=tk.SUNKEN, 
            bd=2
        )
        self.puzzle_frame.pack(fill=tk.BOTH, expand=True)
        
        # ドラッグアンドドロップの設定
        self.puzzle_frame.drop_target_register(DND_FILES)
        self.puzzle_frame.dnd_bind('<<Drop>>', self.on_drop)
        
        # ドロップ指示ラベル
        self.drop_label = tk.Label(
            self.puzzle_frame,
            text="画像をここにドラッグ&ドロップ\nまたは「画像を読み込み」ボタンをクリック",
            bg="lightgray",
            font=("Arial", 12),
            fg="gray"
        )
        self.drop_label.pack(expand=True)
        
        # パズルキャンバス
        self.canvas = tk.Canvas(self.puzzle_frame, bg="white")
        
        # ステータス
        self.status_label = tk.Label(
            self.main_frame, 
            text="画像を読み込んでパズルを開始してください",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=(10, 0))
        
    def change_grid_size(self):
        """グリッドサイズを変更"""
        new_size = self.size_var.get()
        if new_size != self.grid_size:
            if self.is_auto_running:
                self.stop_auto_solve()
            
            self.grid_size = new_size
            self.empty_pos = (self.grid_size-1, self.grid_size-1)
            
            # ソルバーを再初期化
            self.pdb_solver = None
            self.astar_solver = None
            
            # タイトルを更新
            puzzle_names = {3: "8パズル", 4: "15パズル", 5: "24パズル"}
            self.root.title(f"スライディングパズル - 改善版 - {puzzle_names.get(self.grid_size, 'パズル')}")
            
            # 既に画像が読み込まれている場合は画像を保持したまま再構築
            if self.original_image:
                temp_image = self.original_image
                self.reset_tiles_to_solved_state()
                self.original_image = temp_image
                self.update_puzzle_display()
                self.shuffle_button.config(state=tk.NORMAL)
                # 4x4と5x5の場合はAUTOボタンを無効に
                if self.grid_size >= 4:
                    self.auto_button.config(state=tk.DISABLED)
                    self.status_label.config(text=f"パズルサイズが変更されました。{self.grid_size}x{self.grid_size}はAUTO機能は使用できません。シャッフルボタンを押してゲームを開始してください。")
                else:
                    self.status_label.config(text="パズルサイズが変更されました。シャッフルボタンを押してゲームを開始してください。")
            else:
                self.reset_tiles_to_solved_state()
        
    def initialize_solver(self, progress_callback=None):
        """改善されたソルバーを初期化"""
        if self.grid_size >= 3:
            self.pdb_solver = FastPDBSolver(self.grid_size)
            self.pdb_solver.initialize_pdb(progress_callback)
        
        self.astar_solver = FastAStar(self.grid_size, self.pdb_solver)
    
    def on_drop(self, event):
        """ドラッグアンドドロップイベント処理"""
        file_data = event.data
        
        if file_data.startswith('{') and file_data.endswith('}'):
            file_path = file_data.strip('{}')
        else:
            files = []
            current_file = ""
            in_braces = False
            
            i = 0
            while i < len(file_data):
                char = file_data[i]
                if char == '{':
                    in_braces = True
                    current_file += char
                elif char == '}':
                    in_braces = False
                    current_file += char
                    files.append(current_file.strip('{}'))
                    current_file = ""
                elif char == ' ' and not in_braces:
                    if current_file.strip():
                        files.append(current_file.strip())
                        current_file = ""
                else:
                    current_file += char
                i += 1
            
            if current_file.strip():
                files.append(current_file.strip())
            
            if files:
                file_path = files[0]
            else:
                return
        
        if file_path:
            self.load_image_file(file_path)
    
    def load_image(self):
        """ファイルダイアログから画像を読み込み"""
        file_path = filedialog.askopenfilename(
            title="画像ファイルを選択",
            filetypes=[
                ("画像ファイル", "*.jpg *.jpeg *.png *.gif *.bmp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if file_path:
            self.load_image_file(file_path)
    
    def load_image_file(self, file_path):
        """画像ファイルを読み込んで処理"""
        try:
            if self.is_auto_running:
                self.stop_auto_solve()
            
            normalized_path = os.path.normpath(file_path)
            print(f"Loading image: {normalized_path}")
            
            image = Image.open(normalized_path)
            
            # 中心から正方形にクリッピング
            width, height = image.size
            min_size = min(width, height)
            left = (width - min_size) // 2
            top = (height - min_size) // 2
            right = left + min_size
            bottom = top + min_size
            
            square_image = image.crop((left, top, right, bottom))
            self.original_image = square_image
            
            self.reset_tiles_to_solved_state()
            self.update_puzzle_display()
            self.shuffle_button.config(state=tk.NORMAL)
            
            # 4x4と5x5の場合はAUTOボタンを無効にする
            if self.grid_size >= 4:
                self.auto_button.config(state=tk.DISABLED)
                self.status_label.config(text=f"画像が読み込まれました。{self.grid_size}x{self.grid_size}はAUTO機能は使用できません。シャッフルボタンを押してゲームを開始してください。")
            else:
                self.auto_button.config(state=tk.DISABLED)
                self.status_label.config(text="画像が読み込まれました。シャッフルボタンを押してゲームを開始してください。")
            
        except Exception as e:
            error_msg = f"画像の読み込みに失敗しました: {str(e)}"
            print(error_msg)
            messagebox.showerror("エラー", error_msg)
    
    def reset_tiles_to_solved_state(self):
        """タイルを完成状態（解決済み状態）にリセット"""
        self.tiles = []
        for row in range(self.grid_size):
            tile_row = []
            for col in range(self.grid_size):
                if row == self.grid_size - 1 and col == self.grid_size - 1:
                    tile_row.append(0)  # 空のタイル（右下）
                else:
                    tile_row.append(row * self.grid_size + col + 1)
            self.tiles.append(tile_row)
        
        self.empty_pos = (self.grid_size-1, self.grid_size-1)
    
    def update_puzzle_display(self):
        """パズル表示の更新"""
        if not self.original_image:
            return
            
        self.drop_label.pack_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.adjust_puzzle_size()
        self.create_puzzle_tiles()
        self.draw_puzzle()
    
    def adjust_puzzle_size(self):
        """ウィンドウサイズに応じてパズルサイズを調整"""
        self.root.update()
        
        canvas_width = self.puzzle_frame.winfo_width() - 20
        canvas_height = self.puzzle_frame.winfo_height() - 20
        
        max_size = min(canvas_width, canvas_height)
        max_size = max(200, max_size)
        
        self.puzzle_size = max_size
        self.tile_size = self.puzzle_size // self.grid_size
        
        self.canvas.config(width=self.puzzle_size, height=self.puzzle_size)
    
    def create_puzzle_tiles(self):
        """パズルタイルを作成"""
        if not self.original_image:
            return
            
        resized_image = self.original_image.resize((self.puzzle_size, self.puzzle_size), Image.Resampling.LANCZOS)
        
        self.tile_images = []
        for row in range(self.grid_size):
            tile_row = []
            for col in range(self.grid_size):
                if row == self.grid_size - 1 and col == self.grid_size - 1:
                    tile_row.append(None)
                else:
                    left = col * self.tile_size
                    top = row * self.tile_size
                    right = left + self.tile_size
                    bottom = top + self.tile_size
                    
                    tile_image = resized_image.crop((left, top, right, bottom))
                    tile_photo = ImageTk.PhotoImage(tile_image)
                    tile_row.append(tile_photo)
            self.tile_images.append(tile_row)
    
    def draw_puzzle(self):
        """パズルをキャンバスに描画"""
        self.canvas.delete("all")
        
        if not self.tile_images:
            return
            
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.tile_size
                y = row * self.tile_size
                
                if self.tiles[row][col] == 0:
                    self.canvas.create_rectangle(
                        x, y, x + self.tile_size, y + self.tile_size,
                        fill="lightgray", outline="black", width=2
                    )
                else:
                    tile_num = self.tiles[row][col]
                    img_row = (tile_num - 1) // self.grid_size
                    img_col = (tile_num - 1) % self.grid_size
                    
                    if self.tile_images[img_row][img_col]:
                        self.canvas.create_image(
                            x, y, anchor=tk.NW, 
                            image=self.tile_images[img_row][img_col]
                        )
                        
                        self.canvas.create_rectangle(
                            x, y, x + self.tile_size, y + self.tile_size,
                            fill="", outline="black", width=2
                        )
        
        if not self.is_auto_running:
            self.canvas.bind("<Button-1>", self.on_tile_click)
        else:
            self.canvas.unbind("<Button-1>")
    
    def on_tile_click(self, event):
        """タイルクリック処理"""
        if not self.tile_images or self.is_auto_running:
            return
            
        col = event.x // self.tile_size
        row = event.y // self.tile_size
        
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.move_tile(row, col)
    
    def move_tile(self, row, col):
        """タイルを移動"""
        empty_row, empty_col = self.empty_pos
        
        if (abs(row - empty_row) == 1 and col == empty_col) or \
           (abs(col - empty_col) == 1 and row == empty_row):
            
            self.tiles[empty_row][empty_col] = self.tiles[row][col]
            self.tiles[row][col] = 0
            self.empty_pos = (row, col)
            
            self.draw_puzzle()
            
            if self.is_solved():
                self.auto_button.config(state=tk.DISABLED)
                self.status_label.config(text="おめでとうございます！パズルが完成しました！")
                messagebox.showinfo("完成", "パズルが完成しました！おめでとうございます！")
            
            return True
        return False
    
    def is_solved(self):
        """パズルが完成しているかチェック"""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if row == self.grid_size - 1 and col == self.grid_size - 1:
                    if self.tiles[row][col] != 0:
                        return False
                else:
                    expected = row * self.grid_size + col + 1
                    if self.tiles[row][col] != expected:
                        return False
        return True
    
    def shuffle_puzzle(self):
        """パズルをシャッフル - 必ず解ける状態を生成"""
        if not self.tile_images:
            return
        
        if self.is_auto_running:
            self.stop_auto_solve()
        
        # 方法1: 正当な移動のみでシャッフル（確実に解ける）
        self.shuffle_with_valid_moves()
        
        # シャッフル後、解けるかどうかを確認
        if not self.is_solvable():
            print("Warning: Generated unsolvable state, trying alternative shuffle...")
            # 方法2: 転倒数を調整してシャッフル
            self.shuffle_with_inversion_check()
        
        self.draw_puzzle()
        
        # 3x3のみAUTOボタンを有効に、4x4と5x5は無効のまま
        if self.grid_size == 3:
            self.auto_button.config(state=tk.NORMAL)
            self.status_label.config(text="パズルがシャッフルされました（解ける状態保証）。タイルをクリックして移動させてください。")
        else:
            self.auto_button.config(state=tk.DISABLED)
            self.status_label.config(text=f"パズルがシャッフルされました（解ける状態保証）。{self.grid_size}x{self.grid_size}はAUTO機能は使用できません。タイルをクリックして移動させてください。")
    
    def shuffle_with_valid_moves(self):
        """正当な移動のみでシャッフル（100%解ける状態を保証）"""
        shuffle_count = self.grid_size * self.grid_size * 200
        last_move = None
        
        for _ in range(shuffle_count):
            empty_row, empty_col = self.empty_pos
            
            # 移動可能な方向
            possible_moves = []
            if empty_row > 0:
                possible_moves.append((-1, 0, 'up'))
            if empty_row < self.grid_size - 1:
                possible_moves.append((1, 0, 'down'))
            if empty_col > 0:
                possible_moves.append((0, -1, 'left'))
            if empty_col < self.grid_size - 1:
                possible_moves.append((0, 1, 'right'))
            
            # 直前の移動の逆方向を避ける（より効率的なシャッフル）
            if last_move:
                reverse_moves = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
                possible_moves = [move for move in possible_moves if move[2] != reverse_moves[last_move]]
            
            if not possible_moves:
                possible_moves = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
                possible_moves = [move for move in possible_moves 
                                if 0 <= empty_row + move[0] < self.grid_size and 
                                   0 <= empty_col + move[1] < self.grid_size]
            
            # ランダムに移動
            dr, dc, direction = random.choice(possible_moves)
            new_row = empty_row + dr
            new_col = empty_col + dc
            
            self.tiles[empty_row][empty_col] = self.tiles[new_row][new_col]
            self.tiles[new_row][new_col] = 0
            self.empty_pos = (new_row, new_col)
            last_move = direction
    
    def shuffle_with_inversion_check(self):
        """転倒数をチェックして解ける状態を生成"""
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # 完成状態から開始
            self.reset_tiles_to_solved_state()
            
            # ランダムに配置（空白以外）
            tiles = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if not (row == self.grid_size - 1 and col == self.grid_size - 1):
                        tiles.append(row * self.grid_size + col + 1)
            
            random.shuffle(tiles)
            
            # グリッドに配置
            tile_index = 0
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if row == self.grid_size - 1 and col == self.grid_size - 1:
                        self.tiles[row][col] = 0
                        self.empty_pos = (row, col)
                    else:
                        self.tiles[row][col] = tiles[tile_index]
                        tile_index += 1
            
            # 解けるかチェック
            if self.is_solvable():
                print(f"Generated solvable state in {attempt + 1} attempts")
                return
        
        print("Failed to generate solvable state, using valid moves method")
        self.shuffle_with_valid_moves()
    
    def is_solvable(self):
        """パズルが解ける状態かどうかをチェック"""
        # フラットなリストに変換（空白は除く）
        flat_tiles = []
        empty_row = 0
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.tiles[row][col] == 0:
                    empty_row = row
                else:
                    flat_tiles.append(self.tiles[row][col])
        
        # 転倒数を計算
        inversions = 0
        for i in range(len(flat_tiles)):
            for j in range(i + 1, len(flat_tiles)):
                if flat_tiles[i] > flat_tiles[j]:
                    inversions += 1
        
        # 解ける条件をチェック
        if self.grid_size % 2 == 1:
            # 奇数サイズ: 転倒数が偶数なら解ける
            return inversions % 2 == 0
        else:
            # 偶数サイズ: より複雑な条件
            # 空白が下から偶数行目にある場合、転倒数は奇数である必要がある
            # 空白が下から奇数行目にある場合、転倒数は偶数である必要がある
            empty_row_from_bottom = self.grid_size - empty_row
            if empty_row_from_bottom % 2 == 0:
                return inversions % 2 == 1
            else:
                return inversions % 2 == 0
    
    def toggle_auto_solve(self):
        """AUTO解法の開始/停止を切り替え"""
        if self.is_auto_running:
            self.stop_auto_solve()
        else:
            self.start_auto_solve()
    
    def start_auto_solve(self):
        """AUTO解法を開始"""
        if not self.tile_images or self.is_solved():
            return
        
        # 4x4と5x5の場合は実行しない
        if self.grid_size >= 4:
            messagebox.showwarning("制限", f"{self.grid_size}x{self.grid_size}パズルはAUTO機能を使用できません。\n計算時間が非常に長くなるためです。")
            return
        
        self.is_auto_running = True
        self.auto_button.config(text="STOP", bg="red")
        self.shuffle_button.config(state=tk.DISABLED)
        self.status_label.config(text="高速ソルバーを初期化中...")
        
        def solver_worker():
            """別スレッドでソルバーを実行"""
            try:
                # プログレスコールバック
                def progress_callback(message):
                    self.root.after(0, lambda: self.status_label.config(text=message))
                
                # ソルバーを初期化（まだの場合）
                if not self.astar_solver:
                    self.root.after(0, lambda: self.status_label.config(text="Pattern Databaseを構築中..."))
                    self.initialize_solver(progress_callback)
                
                self.root.after(0, lambda: self.status_label.config(text="最適解を探索中..."))
                
                # パズルを解く
                solution = self.astar_solver.solve(self.tiles, progress_callback)
                
                if solution is not None:
                    self.auto_moves = solution
                    self.root.after(0, lambda: self.status_label.config(text=f"解法発見！{len(solution)}手で完成します"))
                    self.root.after(500, self.execute_auto_moves)  # 0.5秒後に実行開始
                else:
                    self.root.after(0, lambda: self.stop_auto_solve())
                    self.root.after(0, lambda: messagebox.showwarning("警告", "解法が見つかりませんでした。"))
                    
            except Exception as e:
                print(f"Solver error: {e}")
                self.root.after(0, lambda: self.stop_auto_solve())
                self.root.after(0, lambda: messagebox.showerror("エラー", f"解法中にエラーが発生しました: {str(e)}"))
        
        # ソルバーを別スレッドで実行
        self.solver_thread = threading.Thread(target=solver_worker, daemon=True)
        self.solver_thread.start()
    
    def stop_auto_solve(self):
        """AUTO解法を停止"""
        self.is_auto_running = False
        self.auto_button.config(text="AUTO (高速)", bg="lightgreen")
        self.shuffle_button.config(state=tk.NORMAL)
        self.auto_moves = []
        
        if self.auto_after_id:
            self.root.after_cancel(self.auto_after_id)
            self.auto_after_id = None
        
        self.draw_puzzle()
        
        if not self.is_solved():
            if self.grid_size == 3:
                self.status_label.config(text="AUTO解法が停止されました。タイルをクリックして移動させてください。")
            else:
                self.status_label.config(text=f"AUTO解法が停止されました。{self.grid_size}x{self.grid_size}はAUTO機能は使用できません。")
    
    def execute_auto_moves(self):
        """AUTO解法の手順を実行"""
        if not self.is_auto_running or not self.auto_moves:
            if self.is_solved():
                self.stop_auto_solve()
                self.status_label.config(text="おめでとうございます！AUTO解法でパズルが完成しました！")
                messagebox.showinfo("完成", "AUTO解法でパズルが完成しました！")
            else:
                self.stop_auto_solve()
            return
        
        # 次の手を実行
        row, col = self.auto_moves.pop(0)
        self.move_tile(row, col)
        
        # 残り手数を表示
        remaining = len(self.auto_moves)
        self.status_label.config(text=f"AUTO解法実行中... 残り{remaining}手")
        
        # 次の手を遅延実行（150ms後）
        self.auto_after_id = self.root.after(150, self.execute_auto_moves)
    
    def reset_puzzle(self):
        """パズルをリセット"""
        if self.is_auto_running:
            self.stop_auto_solve()
        
        self.original_image = None
        self.tile_images = []
        self.shuffle_button.config(state=tk.DISABLED)
        self.auto_button.config(state=tk.DISABLED)
        
        self.reset_tiles_to_solved_state()
        
        self.canvas.pack_forget()
        self.drop_label.pack(expand=True)
        
        self.status_label.config(text="画像を読み込んでパズルを開始してください")
    
    def on_window_resize(self, event):
        """ウィンドウリサイズ時の処理"""
        if event.widget == self.root and self.original_image:
            self.root.after(100, self.delayed_resize)
    
    def delayed_resize(self):
        """遅延リサイズ処理"""
        if self.original_image and self.canvas.winfo_viewable():
            self.update_puzzle_display()

def main():
    # TkinterDnD対応のルートウィンドウを作成
    root = TkinterDnD.Tk()
    app = SlidingPuzzle(root)
    root.mainloop()

if __name__ == "__main__":
    main()