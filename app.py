import sys
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QPainterPath, QKeyEvent
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                               QSplitter, QTextEdit, QComboBox, QGridLayout)

# ==========================================================
# UTILITY
# ==========================================================
def lerp(a, b, t):
    return a + (b - a) * t

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

# ==========================================================
# VISUALIZATION WIDGETS
# ==========================================================

class IsometricMatrixWidget(QWidget):
    """Visualizes Memory as a 3D Isometric Grid of Cubes."""
    def __init__(self, rows, cols, title=""):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.title = title
        
        self.target_memory = np.zeros((rows, cols))
        self.target_read_w1 = np.zeros(rows)
        self.target_read_w2 = np.zeros(rows)
        self.target_write_w = np.zeros(rows)
        
        self.display_memory = np.zeros((rows, cols))
        self.display_read_w1 = np.zeros(rows)
        self.display_read_w2 = np.zeros(rows)
        self.display_write_w = np.zeros(rows)
        
        self.is_backprop = False
        self.backprop_flash = 0.0
        self.setMinimumSize(400, 300)

    def update_targets(self, memory, read_w1, read_w2, write_w):
        self.target_memory = np.clip(memory, -1, 1)
        self.target_read_w1 = read_w1
        self.target_read_w2 = read_w2
        self.target_write_w = write_w

    def trigger_backprop(self):
        self.is_backprop = True
        self.backprop_flash = 1.0

    def tick_animation(self):
        self.display_memory += (self.target_memory - self.display_memory) * 0.15
        self.display_read_w1 += (self.target_read_w1 - self.display_read_w1) * 0.15
        self.display_read_w2 += (self.target_read_w2 - self.display_read_w2) * 0.15
        self.display_write_w += (self.target_write_w - self.display_write_w) * 0.15
        
        if self.is_backprop:
            self.backprop_flash *= 0.9
            if self.backprop_flash < 0.01:
                self.is_backprop = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(24, 24, 32))
        
        painter.setPen(QColor(150, 150, 160))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(10, 15, self.title)

        # 3D Projection Math
        tile_w = 24
        tile_h = 12
        max_height = 40.0
        
        # Center the grid
        grid_w = (self.rows + self.cols) * tile_w / 2
        grid_h = (self.rows + self.cols) * tile_h / 2
        origin_x = self.width() / 2 - grid_w / 2 + tile_w / 2
        origin_y = self.height() / 2 - grid_h / 2 + tile_h / 2 + 20

        max_w_idx = np.argmax(self.display_write_w)
        max_r1_idx = np.argmax(self.display_read_w1)
        max_r2_idx = np.argmax(self.display_read_w2)

        # Draw cells back-to-front (Painter's Algorithm)
        # Sum of indices determines depth. Higher sum = closer to viewer.
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.display_memory[r, c]
                height = abs(val) * max_height
                
                # Base coordinates for the tile
                x = origin_x + (c - r) * tile_w / 2
                y = origin_y + (c + r) * tile_h / 2
                
                # Shift up by height
                z = y - height

                # Colors
                base_color = QColor(20, 20, int(-val * 255), 220) if val < 0 else QColor(int(val * 255), 20, 20, 220)
                left_color = QColor(base_color.red()//2, base_color.green()//2, base_color.blue()//2, 220)
                right_color = QColor(base_color.red()//3, base_color.green()//3, base_color.blue()//3, 220)

                if self.is_backprop:
                    base_color = QColor.lerp(base_color, QColor(255, 165, 0, 255), self.backprop_flash)
                    left_color = QColor.lerp(left_color, QColor(200, 130, 0, 255), self.backprop_flash)
                    right_color = QColor.lerp(right_color, QColor(150, 100, 0, 255), self.backprop_flash)

                # Draw Left Face
                path_left = QPainterPath()
                path_left.moveTo(x, y)
                path_left.lineTo(x, y + tile_h)
                path_left.lineTo(x, z + tile_h)
                path_left.lineTo(x, z)
                painter.setBrush(QBrush(left_color))
                painter.setPen(QPen(QColor(10, 10, 15), 0.5))
                painter.drawPath(path_left)

                # Draw Right Face
                path_right = QPainterPath()
                path_right.moveTo(x, y)
                path_right.lineTo(x + tile_w, y + tile_h/2)
                path_right.lineTo(x + tile_w, z + tile_h/2)
                path_right.lineTo(x, z)
                painter.setBrush(QBrush(right_color))
                painter.drawPath(path_right)

                # Draw Top Face
                path_top = QPainterPath()
                path_top.moveTo(x, z)
                path_top.lineTo(x + tile_w, z + tile_h/2)
                path_top.lineTo(x, z + tile_h)
                path_top.lineTo(x - tile_w, z + tile_h/2)
                painter.setBrush(QBrush(base_color))
                painter.drawPath(path_top)
                
                # Highlight active rows
                if r == max_w_idx and self.display_write_w[max_w_idx] > 0.1:
                    painter.setPen(QPen(QColor(255, 100, 100), 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path_top)
                if r == max_r1_idx and self.display_read_w1[max_r1_idx] > 0.1:
                    painter.setPen(QPen(QColor(100, 255, 100), 1.5))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path_top)
                if r == max_r2_idx and self.display_read_w2[max_r2_idx] > 0.1:
                    painter.setPen(QPen(QColor(100, 255, 255), 1.5))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path_top)


class HistoryWidget(QWidget):
    def __init__(self, title, color):
        super().__init__()
        self.title = title
        self.color = color
        self.history = []
        self.current_step = 0
        self.setMinimumHeight(80)

    def update_data(self, history, step):
        self.history = history
        self.current_step = step
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 28))
        painter.setPen(QColor(150, 150, 160))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(5, 10, self.title)

        if not self.history: return
        margin = 10
        max_steps = len(self.history)
        rows = len(self.history[0]) if max_steps > 0 else 0
        cell_w = (self.width() - margin * 2) / max(1, max_steps)
        cell_h = (self.height() - margin * 2) / max(1, rows)

        for t in range(max_steps):
            for r in range(rows):
                w = self.history[t][r]
                val = int(min(1.0, w) * 255)
                painter.setBrush(QBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), val)))
                painter.setPen(QPen(QColor(30, 30, 35), 0.5))
                painter.drawRect(QRectF(margin + t * cell_w, margin + r * cell_h, cell_w, cell_h))

        if self.current_step > 0:
            x = margin + (self.current_step - 1) * cell_w + cell_w
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawLine(QPointF(x, margin), QPointF(x, self.height() - margin))


class VectorWidget(QWidget):
    def __init__(self, title, color, is_diverging=False):
        super().__init__()
        self.title = title
        self.color = color
        self.is_diverging = is_diverging
        self.target_vec = []
        self.display_vec = []
        self.active_idx = -1
        self.is_active = False
        self.setMinimumHeight(40)

    def update_data(self, vec, active_idx=-1, is_active=False):
        self.target_vec = vec if isinstance(vec, list) else list(vec)
        self.active_idx = active_idx
        self.is_active = is_active
        if len(self.display_vec) != len(self.target_vec):
            self.display_vec = list(self.target_vec)

    def tick_animation(self):
        if not self.target_vec: return
        for i in range(len(self.target_vec)):
            if i < len(self.display_vec):
                self.display_vec[i] = lerp(self.display_vec[i], self.target_vec[i], 0.2)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        bg_color = QColor(35, 35, 45) if self.is_active else QColor(30, 30, 38)
        painter.fillRect(self.rect(), bg_color)
        border_color = self.color if self.is_active else QColor(60, 60, 70)
        painter.setPen(QPen(border_color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3, 3)
        
        painter.setPen(QColor(150, 150, 160))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(5, 12, self.title)

        if not self.display_vec: return
        margin = 5
        cell_w = (self.width() - margin * 2) / len(self.display_vec)
        cell_h = 16
        start_y = 16

        for i, val in enumerate(self.display_vec):
            if i == self.active_idx:
                painter.setBrush(QBrush(QColor(255, 255, 0, 200)))
            else:
                if self.is_diverging:
                    color = QColor(20, 100, 255, 200) if val < 0 else QColor(255, 50, 50, 200)
                else:
                    alpha = int(min(1.0, abs(val)) * 255)
                    color = QColor(self.color.red(), self.color.green(), self.color.blue(), alpha)
                painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawRoundedRect(QRectF(margin + i * cell_w, start_y, cell_w, cell_h), 2, 2)


class GateWidget(QWidget):
    def __init__(self, title, color, max_val=1.0):
        super().__init__()
        self.title = title
        self.color = color
        self.target_val = 0.0
        self.display_val = 0.0
        self.max_val = max_val
        self.is_active = False
        self.setMinimumHeight(40)

    def update_data(self, val, is_active=False):
        self.target_val = float(val)

    def tick_animation(self):
        self.display_val += (self.target_val - self.display_val) * 0.2
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        bg_color = QColor(35, 35, 45) if self.is_active else QColor(30, 30, 38)
        painter.fillRect(self.rect(), bg_color)
        border_color = self.color if self.is_active else QColor(60, 60,70)
        painter.setPen(QPen(border_color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3, 3)
        
        painter.setPen(QColor(150, 150, 160))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(5, 12, self.title)
        
        track_rect = QRectF(5, 20, self.width()-10, 12)
        painter.setBrush(QBrush(QColor(20, 20, 20)))
        painter.setPen(QPen(QColor(50, 50, 50)))
        painter.drawRoundedRect(track_rect, 6, 6)
        
        pct = clamp(self.display_val / self.max_val, 0.0, 1.0)
        fill_w = (self.width()-10) * pct
        fill_rect = QRectF(5, 20, fill_w, 12)
        painter.setBrush(QBrush(self.color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(fill_rect, 6, 6)


class LossCurveWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.losses = [2.0]
        self.max_points = 50
        self.setMinimumHeight(80)

    def add_loss(self, success):
        last = self.losses[-1]
        noise = np.random.normal(0, 0.05)
        drop = 0.1 if success else 0.02
        new_loss = max(0.01, last - drop + noise)
        self.losses.append(new_loss)
        if len(self.losses) > self.max_points:
            self.losses.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 32))
        painter.setPen(QColor(150, 150, 160))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(5, 10, "Simulated Training Loss")
        
        if len(self.losses) < 2: return
        margin = 10
        w = self.width() - margin * 2
        h = self.height() - margin * 2 - 10
        
        path = QPainterPath()
        for i, loss in enumerate(self.losses):
            x = margin + (i / (self.max_points - 1)) * w
            y = margin + 10 + (loss / 2.0) * h
            if i == 0: path.moveTo(x, y)
            else: path.lineTo(x, y)
                
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawPath(path)


class WeightDynamicsChart(QWidget):
    """Plots the attention weight of a specific row over time."""
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.history = [] # List of weight arrays
        self.target_row = 0
        self.setMinimumHeight(100)

    def update_data(self, history, target_row):
        self.history = history
        self.target_row = target_row
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 32))
        painter.setPen(QColor(150, 150, 160))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(5, 10, self.title)
        
        if len(self.history) < 2: return
        margin = 10
        w = self.width() - margin * 2
        h = self.height() - margin * 2 - 10
        
        path = QPainterPath()
        for t, weights in enumerate(self.history):
            if self.target_row < len(weights):
                val = weights[self.target_row]
                x = margin + (t / (len(self.history)-1)) * w
                y = margin + 10 + (1.0 - val) * h
                if t == 0: path.moveTo(x, y)
                else: path.lineTo(x, y)
                
        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.drawPath(path)


class ControllerTopologyWidget(QWidget):
    """Visualizes the Controller's internal MLP structure."""
    def __init__(self):
        super().__init__()
        self.hidden_state = np.zeros(6)
        self.input_val = 0.0
        self.output_val = 0.0
        self.is_active = False
        self.pulse = 0.0
        self.setMinimumHeight(120)

    def update_data(self, hidden, inp, outp, is_active):
        self.hidden_state = hidden
        self.input_val = inp
        self.output_val = outp
        self.is_active = is_active
        if is_active:
            self.pulse = 1.0

    def tick_animation(self):
        self.pulse *= 0.9
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 28))
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QColor(150, 150, 160))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(5, 10, "Controller Topology (MLP)")

        # Node positions
        in_pos = QPointF(20, 60)
        out_pos = QPointF(self.width()-20, 60)
        hid_pos = [QPointF(self.width()/2, 20 + i*16) for i in range(6)]

        # Draw edges
        edge_col = QColor(255, 165, 0, 100) if self.is_active else QColor(60, 60, 60)
        painter.setPen(QPen(edge_col, 1))
        for h in hid_pos:
            painter.drawLine(in_pos, h)
            painter.drawLine(h, out_pos)

        # Draw nodes
        # Input
        in_col = QColor(255, 200, 0) if self.input_val > 0 else QColor(80, 60, 0)
        painter.setBrush(QBrush(in_col))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(in_pos, 8, 8)

        # Hidden
        for i, h in enumerate(hid_pos):
            val = self.hidden_state[i] if i < len(self.hidden_state) else 0
            col = QColor(20, 100, 255) if val < 0 else QColor(255, 50, 50)
            alpha = int(min(1.0, abs(val)) * 255)
            col.setAlpha(alpha)
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.drawEllipse(h, 6, 6)

        # Output
        out_col = QColor(0, 150, 255) if self.output_val > 0 else QColor(0, 40, 80)
        painter.setBrush(QBrush(out_col))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(out_pos, 8, 8)

        # Pulse effect
        if self.pulse > 0.01:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 165, 0, int(self.pulse * 255)), 2))
            painter.drawEllipse(in_pos, 8 + (1-self.pulse)*10, 8 + (1-self.pulse)*10)


class ArchitectureWidget(QWidget):
    """Visualizes data flow with animated particles and backprop support."""
    def __init__(self):
        super().__init__()
        self.active_flows = []
        self.is_backward = False
        self.particles = []
        self.particle_spawn_timer = 0
        self.setMinimumHeight(160)
        
    def set_active_flows(self, flows, is_backward=False):
        self.active_flows = flows
        self.is_backward = is_backward
        if is_backward:
            self.active_flows = [f.replace("->", "<-") for f in flows]
        self.update()

    def tick_animation(self):
        self.particle_spawn_timer += 1
        if self.particle_spawn_timer > 8:
            self.particle_spawn_timer = 0
            for flow_id in self.active_flows:
                self.particles.append({"flow_id": flow_id, "progress": 0.0 if not self.is_backward else 1.0})
        
        updated_particles = []
        for p in self.particles:
            if self.is_backward:
                p["progress"] -= 0.05
                if p["progress"] >= 0.0: updated_particles.append(p)
            else:
                p["progress"] += 0.05
                if p["progress"] <= 1.0: updated_particles.append(p)
        self.particles = updated_particles
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 28))
        
        nodes = {
            "Input": QRectF(20, 60, 80, 30),
            "Controller": QRectF(130, 60, 100, 30),
            "Params": QRectF(130, 10, 100, 30),
            "WriteHead": QRectF(270, 100, 90, 30),
            "ReadHead1": QRectF(270, 50, 90, 30),
            "ReadHead2": QRectF(270, 0, 90, 30),
            "Memory": QRectF(400, 60, 80, 30),
            "Output": QRectF(130, 110, 100, 30)
        }
        
        edge_map = {
            "in->ctrl": ("Input", "Controller", "line"),
            "ctrl->params": ("Controller", "Params", "line"),
            "params->wh": ("Params", "WriteHead", "line"),
            "params->rh1": ("Params", "ReadHead1", "line"),
            "params->rh2": ("Params", "ReadHead2", "line"),
            "wh->mem": ("WriteHead", "Memory", "line"),
            "mem->rh1": ("Memory", "ReadHead1", "line"),
            "mem->rh2": ("Memory", "ReadHead2", "line"),
            "rh1->rh2": ("ReadHead1", "ReadHead2", "curve"),
            "ctrl->out": ("Controller", "Output", "line")
        }
        
        painter.setFont(QFont("Segoe UI", 7))
        edge_paths = {}
        for flow_id, (src, dst, edge_type) in edge_map.items():
            r1 = nodes[src]; r2 = nodes[dst]
            start = QPointF(r1.right(), r1.center().y())
            end = QPointF(r2.left(), r2.center().y())
            if self.is_backward: start, end = end, start
            
            path = QPainterPath()
            if edge_type == "curve":
                if not self.is_backward:
                    start = QPointF(r1.right(), r1.top()); end = QPointF(r2.right(), r2.bottom())
                    path.moveTo(start); path.cubicTo(QPointF(start.x()+30, start.y()), QPointF(end.x()+30, end.y()), end)
                else:
                    start = QPointF(r2.right(), r2.bottom()); end = QPointF(r1.right(), r1.top())
                    path.moveTo(start); path.cubicTo(QPointF(start.x()+30, start.y()), QPointF(end.x()+30, end.y()), end)
            else:
                path.moveTo(start); path.lineTo(end)
                
            edge_paths[flow_id] = path
            color = QColor(255, 165, 0, 255) if self.is_backward else QColor(100, 255, 100, 255)
            inactive_color = QColor(80, 60, 30) if self.is_backward else QColor(80, 80, 80)
            final_color = color if flow_id in self.active_flows else inactive_color
            painter.setPen(QPen(final_color, 2 if flow_id in self.active_flows else 1))
            painter.drawPath(path)

        for p in self.particles:
            if p["flow_id"] in edge_paths:
                path = edge_paths[p["flow_id"]]
                t = clamp(p["progress"], 0.0, 1.0)
                if path.elementCount() > 0:
                    pt = path.pointAtPercent(t)
                    p_color = QColor(255, 200, 100, 255) if self.is_backward else QColor(255, 255, 255, 255)
                    painter.setBrush(QBrush(p_color))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(pt, 3, 3)

        for name, rect in nodes.items():
            painter.setBrush(QBrush(QColor(40, 40, 48)))
            painter.setPen(QPen(QColor(100, 100, 120), 1))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(rect, Qt.AlignCenter, name)


# ==========================================================
# NTM SIMULATOR
# ==========================================================

class NTMSimulator:
    def __init__(self, mem_rows=10, mem_cols=8, seq_length=4, repeat=2, task="copy"):
        self.mem_rows = mem_rows
        self.mem_cols = mem_cols
        self.seq_length = seq_length
        self.repeat = repeat
        self.task = task
        self.reset()

    def reset(self):
        self.memory = np.random.uniform(-0.05, 0.05, (self.mem_rows, self.mem_cols))
        
        self.read_w1 = np.zeros(self.mem_rows)
        self.read_w2 = np.zeros(self.mem_rows)
        self.prev_read_w1 = np.zeros(self.mem_rows)
        self.write_w = np.zeros(self.mem_rows)
        self.prev_write_w = np.zeros(self.mem_rows)
        
        self.read1_hist = []
        self.read2_hist = []
        self.write_hist = []
        
        if self.task == "associative_recall":
            self.pairs = []
            for _ in range(self.seq_length):
                k = np.random.randint(1, 10); v = np.random.randint(0, 2)
                self.pairs.append((k, v))
            self.query_idx = np.random.randint(0, self.seq_length)
        elif self.task == "priority_sort":
            self.input_seq = np.random.randint(0, 2, self.seq_length)
            self.priorities = np.random.rand(self.seq_length)
            self.sorted_indices = np.argsort(self.priorities)[::-1]
        elif self.task == "dynamic_copy":
            self.input_seq = np.random.randint(0, 2, self.seq_length)
            self.write_targets = np.random.choice(self.mem_rows, self.seq_length, replace=False)
            self.read_targets = np.sort(self.write_targets) # Read back sequentially
        else:
            self.input_seq = np.random.randint(0, 2, self.seq_length)
            
        self.output_seq = []
        
        self.phase = "WRITE"
        self.step_idx = 0
        self.repeat_count = 0
        
        self.current_input = 0
        self.read_vector1 = np.zeros(self.mem_cols)
        self.read_vector2 = np.zeros(self.mem_cols)
        self.current_output = 0
        self.hidden_state = np.zeros(6)
        
        self.erase_vector = np.zeros(self.mem_cols)
        self.add_vector = np.zeros(self.mem_cols)
        self.key_vector = np.zeros(self.mem_cols)
        
        self.gate_r1 = 0.0
        self.gate_w = 0.0
        self.shift_kernel_w = np.zeros(3)
        self.shift_kernel_r = np.zeros(3)
        self.gamma_w = 1.0
        self.gamma_r = 1.0

    def _get_fuzzy_weights(self, target_row, sharpness=1.5):
        rows = np.arange(self.mem_rows)
        weights = np.exp(-0.5 * ((rows - target_row) / sharpness)**2)
        return weights / np.sum(weights)

    def _content_address(self, key):
        sim = np.dot(self.memory, key)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        return sim / np.sum(sim)

    def _convolve_shift(self, weights, shift_vec):
        shifted = np.zeros_like(weights)
        shifted += np.roll(weights, 1) * shift_vec[0]
        shifted += weights * shift_vec[1]
        shifted += np.roll(weights, -1) * shift_vec[2]
        return shifted

    def _sharpen(self, weights, gamma):
        w = weights ** gamma
        return w / (np.sum(w) + 1e-8)

    def step(self):
        if self.phase == "WRITE":
            if self.step_idx < self.seq_length:
                if self.task == "dynamic_copy":
                    target_row = self.write_targets[self.step_idx]
                else:
                    target_row = self.step_idx
                    
                self.key_vector = np.zeros(self.mem_cols)
                self.key_vector[target_row] = 1.0
                content_w = self._content_address(self.key_vector)
                
                self.gate_w = 1.0
                self.prev_write_w = self.write_w.copy()
                self.write_w = self.gate_w * content_w + (1 - self.gate_w) * self.prev_write_w
                
                self.shift_kernel_w = np.array([0.0, 1.0, 0.0])
                self.write_w = self._convolve_shift(self.write_w, self.shift_kernel_w)
                
                self.gamma_w = 2.0
                self.write_w = self._sharpen(self.write_w, self.gamma_w)
                
                self.read_w1 = np.zeros(self.mem_rows)
                self.read_w2 = np.zeros(self.mem_rows)
                
                if self.task == "associative_recall":
                    k, v = self.pairs[self.step_idx]
                    self.add_vector = np.zeros(self.mem_cols)
                    self.add_vector[k % self.mem_cols] = 1.0
                    self.add_vector[(v + 5) % self.mem_cols] = 1.0 if v == 1 else -1.0
                    self.current_input = k
                else:
                    self.current_input = self.input_seq[self.step_idx]
                    self.add_vector = np.full(self.mem_cols, self.current_input * 2 - 1)
                
                self.erase_vector = np.ones(self.mem_cols)
                
                w_matrix = np.outer(self.write_w, self.erase_vector)
                self.memory = self.memory * (1 - w_matrix)
                self.memory += np.outer(self.write_w, self.add_vector)
                
                self.hidden_state = np.tanh(np.random.uniform(-1, 1, 6))
                
                self.read_vector1 = np.zeros(self.mem_cols)
                self.read_vector2 = np.zeros(self.mem_cols)
                self.step_idx += 1
                return "WRITE", f"Writing pair to Mem[{target_row}]"
            else:
                self.phase = "READ"
                self.step_idx = 0
                self.write_w = np.zeros(self.mem_rows)
                return "TRANSITION", "Switching to Read Phase..."

        elif self.phase == "READ":
            if self.step_idx < self.seq_length:
                if self.task == "associative_recall":
                    q_key, _ = self.pairs[self.query_idx]
                    self.key_vector = np.zeros(self.mem_cols)
                    self.key_vector[q_key % self.mem_cols] = 1.0
                    content_w = self._content_address(self.key_vector)
                    self.gate_r1 = 1.0
                    self.read_w1 = self.gate_r1 * content_w + (1 - self.gate_r1) * self.prev_read_w1
                    self.shift_kernel_r = np.array([0.0, 0.0, 1.0])
                    self.read_w2 = self._convolve_shift(self.read_w1, self.shift_kernel_r)
                    self.gamma_r = 2.0
                    self.read_w1 = self._sharpen(self.read_w1, self.gamma_r)
                    self.read_w2 = self._sharpen(self.read_w2, self.gamma_r)
                    self.read_vector1 = np.dot(self.read_w1, self.memory)
                    self.read_vector2 = np.dot(self.read_w2, self.memory)
                    self.current_output = 1 if self.read_vector2[(5) % self.mem_cols] > 0 else 0
                    self.output_seq.append(self.current_output)
                    self.query_idx = (self.query_idx + 1) % self.seq_length
                    target_row = np.argmax(self.read_w1)
                elif self.task == "priority_sort":
                    target_row = self.sorted_indices[self.step_idx]
                    self.key_vector = self.memory[target_row].copy()
                    content_w = self._content_address(self.key_vector)
                    self.gate_r1 = 1.0
                    self.read_w1 = self.gate_r1 * content_w + (1 - self.gate_r1) * self.prev_read_w1
                    self.shift_kernel_r = np.array([0.0, 1.0, 0.0])
                    self.read_w2 = self._convolve_shift(self.read_w1, self.shift_kernel_r)
                    self.gamma_r = 5.0
                    self.read_w1 = self._sharpen(self.read_w1, self.gamma_r)
                    self.read_vector1 = np.dot(self.read_w1, self.memory)
                    self.current_output = 1 if np.mean(self.read_vector1) > 0 else 0
                    self.output_seq.append(self.current_output)
                elif self.task == "dynamic_copy":
                    target_row = self.read_targets[self.step_idx]
                    self.read_w1 = self._get_fuzzy_weights(target_row)
                    self.read_w2 = np.zeros(self.mem_rows)
                    self.read_vector1 = np.dot(self.read_w1, self.memory)
                    self.current_output = 1 if np.mean(self.read_vector1) > 0 else 0
                    self.output_seq.append(self.current_output)
                else:
                    target_row = self.step_idx
                    self.read_w1 = self._get_fuzzy_weights(target_row)
                    self.read_w2 = np.zeros(self.mem_rows)
                    self.read_vector1 = np.dot(self.read_w1, self.memory)
                    self.current_output = 1 if np.mean(self.read_vector1) > 0 else 0
                    self.output_seq.append(self.current_output)
                
                self.hidden_state = np.tanh(np.random.uniform(-1, 1, 6) + (0.5 if self.current_output else -0.5))
                
                self.erase_vector = np.zeros(self.mem_cols)
                self.add_vector = np.zeros(self.mem_cols)
                self.step_idx += 1
                return "READ", f"Reading bit {self.current_output} from Mem[{target_row}]"
            else:
                self.phase = "DONE"
                return "DONE", "Task Complete!"
        return "IDLE", "Idle"

    def update_history(self):
        self.read1_hist.append(self.read_w1.copy())
        self.read2_hist.append(self.read_w2.copy())
        self.write_hist.append(self.write_w.copy())


# ==========================================================
# MAIN APPLICATION
# ==========================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Turing Machine - Ultimate Research Dashboard v6.0")
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { color: #E0E0E0; font-family: 'Segoe UI'; }
            QGroupBox { 
                border: 1px solid #333; border-radius: 6px; 
                margin-top: 12px; padding: 10px;
                font-weight: bold; color: #888;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { 
                background-color: #2A2A2A; color: #FFF; 
                border: 1px solid #444; border-radius: 4px; 
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #3A3A3A; border: 1px solid #4CAF50; }
            QComboBox { background-color: #2A2A2A; border: 1px solid #444; border-radius: 4px; padding: 4px; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 4px; background: #2A2A2A; }
            QSlider::handle:horizontal { background: #4CAF50; width: 12px; margin: -5px 0; border-radius: 6px; }
            QTextEdit { background-color: #1A1A1A; color: #4CAF50; border: 1px solid #333; font-family: Consolas; }
            QScrollBar:vertical { border: none; background: #1A1A1A; width: 8px; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; }
            QToolTip { background-color: #2A2A2A; color: #FFF; border: 1px solid #4CAF50; }
        """)

        self.ntm = NTMSimulator(mem_rows=10, mem_cols=8, seq_length=4, repeat=2, task="copy")
        self.is_playing = True
        self.is_backprop = False
        self.delay_ms = 600
        self.epoch_count = 0

        self.init_ui()
        self.setup_timers()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)

        # --- Top Control Bar ---
        controls_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("⏸ Pause")
        self.btn_play.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.btn_play)
        
        self.btn_step = QPushButton("⏭ Step")
        self.btn_step.clicked.connect(self.step_once)
        controls_layout.addWidget(self.btn_step)
        
        self.btn_reset = QPushButton("↻ Reset Task")
        self.btn_reset.clicked.connect(self.reset_ntm)
        controls_layout.addWidget(self.btn_reset)
        
        self.btn_backprop = QPushButton("🔄 Trigger Backprop")
        self.btn_backprop.clicked.connect(self.trigger_backprop)
        controls_layout.addWidget(self.btn_backprop)

        controls_layout.addStretch()
        
        controls_layout.addWidget(QLabel("Task:"))
        self.combo_task = QComboBox()
        self.combo_task.addItems(["copy", "repeat_copy", "associative_recall", "priority_sort", "dynamic_copy"])
        self.combo_task.currentTextChanged.connect(self.change_config)
        controls_layout.addWidget(self.combo_task)

        controls_layout.addWidget(QLabel("Seq Len:"))
        self.slider_seq = QSlider(Qt.Horizontal)
        self.slider_seq.setMinimum(2); self.slider_seq.setMaximum(8); self.slider_seq.setValue(4)
        self.slider_seq.setFixedWidth(60)
        self.slider_seq.valueChanged.connect(self.change_config)
        controls_layout.addWidget(self.slider_seq)
        self.lbl_seq_val = QLabel("4")
        controls_layout.addWidget(self.lbl_seq_val)

        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setMinimum(100); self.slider_speed.setMaximum(1500); self.slider_speed.setValue(self.delay_ms)
        self.slider_speed.setFixedWidth(80)
        self.slider_speed.valueChanged.connect(self.change_speed)
        controls_layout.addWidget(self.slider_speed)

        main_layout.addLayout(controls_layout)

        # --- Status Bar ---
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #4CAF50;")
        main_layout.addWidget(self.status_label)

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        
        # Left Column
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        mem_group = QGroupBox("3D Isometric Memory Matrix & Dual Read Heads")
        mem_layout = QVBoxLayout(mem_group)
        self.memory_widget = IsometricMatrixWidget(self.ntm.mem_rows, self.ntm.mem_cols, "Red=Write, Green=Read1, Cyan=Read2")
        mem_layout.addWidget(self.memory_widget, stretch=3)
        left_layout.addWidget(mem_group, stretch=2)

        hist_group = QGroupBox("Attention History & Dynamics Over Time")
        hist_layout = QVBoxLayout(hist_group)
        self.write_hist_widget = HistoryWidget("Write Head History", QColor(255, 100, 100))
        hist_layout.addWidget(self.write_hist_widget)
        self.read1_hist_widget = HistoryWidget("Read Head 1 History", QColor(100, 255, 100))
        hist_layout.addWidget(self.read1_hist_widget)
        self.read2_hist_widget = HistoryWidget("Read Head 2 History", QColor(100, 255, 255))
        hist_layout.addWidget(self.read2_hist_widget)
        
        self.weight_dyn_widget = WeightDynamicsChart("Read Head 1 Weight (Row 0) Over Time")
        hist_layout.addWidget(self.weight_dyn_widget)
        left_layout.addWidget(hist_group, stretch=1)
        
        splitter.addWidget(left_widget)

        # Right Column
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Architecture & Topology
        arch_top_layout = QHBoxLayout()
        
        arch_group = QGroupBox("NTM Architecture & Animated Data Flow")
        arch_layout = QVBoxLayout(arch_group)
        self.arch_widget = ArchitectureWidget()
        arch_layout.addWidget(self.arch_widget)
        arch_top_layout.addWidget(arch_group)

        topo_group = QGroupBox("Controller Topology (MLP)")
        topo_layout = QVBoxLayout(topo_group)
        self.topo_widget = ControllerTopologyWidget()
        topo_layout.addWidget(self.topo_widget)
        arch_top_layout.addWidget(topo_group)

        right_layout.addLayout(arch_top_layout, stretch=0)

        # Equation Dashboard
        eq_group = QGroupBox("Active Mathematical Operations")
        eq_layout = QVBoxLayout(eq_group)
        self.lbl_equations = QLabel("Waiting for step...")
        self.lbl_equations.setStyleSheet("font-family: Consolas; font-size: 11px; color: #AAA;")
        self.lbl_equations.setWordWrap(True)
        eq_layout.addWidget(self.lbl_equations)
        right_layout.addWidget(eq_group, stretch=0)

        # Controller Vectors Grid
        ctrl_group = QGroupBox("Controller Interface & Addressing Parameters")
        ctrl_layout = QGridLayout(ctrl_group)
        ctrl_layout.setSpacing(5)
        
        self.input_widget = VectorWidget("Input (x_t)", QColor(255, 200, 0))
        ctrl_layout.addWidget(self.input_widget, 0, 0)
        self.output_widget = VectorWidget("Output (y_t)", QColor(0, 150, 255))
        ctrl_layout.addWidget(self.output_widget, 0, 1)
        
        self.hidden_widget = VectorWidget("Controller Hidden State (h_t)", QColor(255, 150, 0), is_diverging=True)
        ctrl_layout.addWidget(self.hidden_widget, 1, 0, 1, 2)
        
        self.key_widget = VectorWidget("Key Vector (k_t)", QColor(200, 100, 255))
        ctrl_layout.addWidget(self.key_widget, 2, 0)
        self.add_widget = VectorWidget("Add Vector (a_t)", QColor(150, 150, 255))
        ctrl_layout.addWidget(self.add_widget, 2, 1)
        
        self.read_vec1_widget = VectorWidget("Read Vec 1 (r_t)", QColor(100, 255, 100))
        ctrl_layout.addWidget(self.read_vec1_widget, 3, 0)
        self.read_vec2_widget = VectorWidget("Read Vec 2 (r_t)", QColor(100, 255, 255))
        ctrl_layout.addWidget(self.read_vec2_widget, 3, 1)
        
        self.erase_widget = VectorWidget("Erase Vector (e_t)", QColor(255, 100, 100))
        ctrl_layout.addWidget(self.erase_widget, 4, 0)
        
        self.gate_w_widget = GateWidget("Write Gate (g_t)", QColor(255, 100, 100))
        ctrl_layout.addWidget(self.gate_w_widget, 4, 1)
        self.gate_r1_widget = GateWidget("Read Gate (g_t)", QColor(100, 255, 100))
        ctrl_layout.addWidget(self.gate_r1_widget, 5, 0)
        self.gamma_w_widget = GateWidget("Write Sharpening (γ_t)", QColor(255, 200, 0), max_val=10.0)
        ctrl_layout.addWidget(self.gamma_w_widget, 5, 1)
        
        self.shift_w_widget = VectorWidget("Write Shift (s_t)", QColor(255, 255, 100))
        ctrl_layout.addWidget(self.shift_w_widget, 6, 0)
        self.shift_r_widget = VectorWidget("Read Shift (s_t)", QColor(255, 255, 100))
        ctrl_layout.addWidget(self.shift_r_widget, 6, 1)
        
        right_layout.addWidget(ctrl_group, stretch=2)

        # Bottom Right: Log & Loss
        bottom_right_layout = QHBoxLayout()
        
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout(log_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)
        bottom_right_layout.addWidget(log_group, stretch=2)
        
        loss_group = QGroupBox("Training Meta-Data")
        loss_layout = QVBoxLayout(loss_group)
        self.loss_widget = LossCurveWidget()
        loss_layout.addWidget(self.loss_widget)
        self.lbl_epoch = QLabel("Epoch: 0")
        loss_layout.addWidget(self.lbl_epoch)
        bottom_right_layout.addWidget(loss_group, stretch=1)

        right_layout.addLayout(bottom_right_layout, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])
        main_layout.addWidget(splitter, stretch=1)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Right:
            self.step_once()
        elif event.key() == Qt.Key_R:
            self.reset_ntm()
        elif event.key() == Qt.Key_B:
            self.trigger_backprop()
        else:
            super().keyPressEvent(event)

    def setup_timers(self):
        self.logic_timer = QTimer(self)
        self.logic_timer.timeout.connect(self.update_ntm)
        self.logic_timer.start(self.delay_ms)
        
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.tick_ui)
        self.render_timer.start(16)

    def tick_ui(self):
        self.memory_widget.tick_animation()
        self.arch_widget.tick_animation()
        self.topo_widget.tick_animation()
        widgets = [self.input_widget, self.output_widget, self.hidden_widget, self.read_vec1_widget, 
                   self.read_vec2_widget, self.key_widget, self.erase_widget, 
                   self.add_widget, self.shift_w_widget, self.shift_r_widget, 
                   self.gate_w_widget, self.gate_r1_widget, self.gamma_w_widget]
        for w in widgets:
            w.tick_animation()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ Pause")
            self.logic_timer.start(self.delay_ms)
        else:
            self.btn_play.setText("▶ Play")
            self.logic_timer.stop()

    def step_once(self):
        self.logic_timer.stop()
        self.is_playing = False
        self.btn_play.setText("▶ Play")
        self.update_ntm()

    def reset_ntm(self):
        self.ntm.task = self.combo_task.currentText()
        self.ntm.seq_length = self.slider_seq.value()
        self.ntm.reset()
        self.log_console.clear()
        self.memory_widget.display_memory = np.zeros_like(self.memory_widget.target_memory)
        self.update_ntm()

    def change_config(self):
        self.lbl_seq_val.setText(str(self.slider_seq.value()))
        self.reset_ntm()

    def change_speed(self, val):
        self.delay_ms = val
        if self.is_playing:
            self.logic_timer.start(self.delay_ms)

    def trigger_backprop(self):
        self.is_backprop = True
        self.memory_widget.trigger_backprop()
        active_flows = ["in->ctrl", "ctrl->params", "params->wh", "wh->mem"] if self.ntm.phase == "WRITE" else ["mem->rh1", "rh1->rh2", "rh1->ctrl", "rh2->ctrl", "ctrl->out"]
        self.arch_widget.set_active_flows(active_flows, is_backward=True)
        self.status_label.setText("Backpropagating Gradients...")
        self.log("⚠ BACKWARD PASS: Computing Gradients (BPTT)...")
        QTimer.singleShot(1500, self.end_backprop)

    def end_backprop(self):
        self.is_backprop = False
        self.update_ntm()

    def log(self, text):
        self.log_console.append(text)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def update_ntm(self):
        if self.is_backprop: return
        
        phase, status_msg = self.ntm.step()
        
        if phase == "DONE":
            self.status_label.setText("Task Complete!")
            self.log(f"[{self.ntm.step_idx}] DONE.")
            
            self.epoch_count += 1
            self.lbl_epoch.setText(f"Epoch: {self.epoch_count}")
            expected = self.ntm.input_seq
            if self.ntm.task == "repeat_copy": expected = np.tile(self.ntm.input_seq, self.ntm.repeat)
            elif self.ntm.task == "associative_recall": expected = [v for k, v in self.ntm.pairs]
            elif self.ntm.task == "priority_sort": expected = self.ntm.input_seq[self.ntm.sorted_indices]
            elif self.ntm.task == "dynamic_copy": expected = self.ntm.input_seq
            success = np.array_equal(expected, self.ntm.output_seq)
            self.loss_widget.add_loss(success)
            
            self.logic_timer.stop()
            QTimer.singleShot(2500, self.reset_ntm)
            return

        if phase == "TRANSITION":
            self.status_label.setText(status_msg)
            self.log(f"--- {status_msg} ---")
            QTimer.singleShot(100, self.update_ntm)
            return
            
        self.ntm.update_history()
        
        active_flows = []
        eq_text = ""
        is_writing = phase == "WRITE"
        if is_writing:
            self.status_label.setText(f"Phase: WRITE | Step: {self.ntm.step_idx}/{self.ntm.seq_length}")
            self.log(f"[t={len(self.ntm.write_hist)}] WRITE: M_t = M_(t-1) * (1 - w_t ⊗ e_t) + w_t ⊗ a_t")
            active_flows = ["in->ctrl", "ctrl->params", "params->wh", "wh->mem"]
            eq_text = (
                "<b>1. Content Addressing:</b> C(M, k_t)<br>"
                "<b>2. Interpolation:</b> w_t = g_t * C + (1-g_t) * w_{t-1}<br>"
                "<b>3. Shift:</b> w_t = Σ s_t(i) * w_t(i)<br>"
                "<b>4. Sharpen:</b> w_t = w_t^γ / Σ w_t^γ<br>"
                "<b>5. Memory Update:</b> M_t = M_{t-1} ⊙ (1 - w_t ⊗ e_t) + w_t ⊗ a_t"
            )
        else:
            self.status_label.setText(f"Phase: READ | Step: {self.ntm.step_idx}/{self.ntm.seq_length}")
            self.log(f"[t={len(self.ntm.read1_hist)}] READ: w_t = Sharpen(Shift(g_t * C(M,k) + (1-g_t)*w_{t-1})) | r_t = w_t^T * M_t")
            active_flows = ["mem->rh1", "rh1->rh2", "rh1->ctrl", "rh2->ctrl", "ctrl->out"]
            eq_text = (
                "<b>1. Content Addressing:</b> C(M, k_t)<br>"
                "<b>2. Interpolation:</b> w_t = g_t * C + (1-g_t) * w_{t-1}<br>"
                "<b>3. Shift:</b> w_t = Σ s_t(i) * w_t(i)<br>"
                "<b>4. Sharpen:</b> w_t = w_t^γ / Σ w_t^γ<br>"
                "<b>5. Read Vector:</b> r_t = w_t^T * M_t"
            )
        
        self.lbl_equations.setText(eq_text)
        self.arch_widget.set_active_flows(active_flows, is_backward=False)
        self.topo_widget.update_data(self.ntm.hidden_state, self.ntm.current_input, self.ntm.current_output, is_active=True)

        self.memory_widget.update_targets(self.ntm.memory, self.ntm.read_w1, self.ntm.read_w2, self.ntm.write_w)
        self.write_hist_widget.update_data(self.ntm.write_hist, len(self.ntm.write_hist))
        self.read1_hist_widget.update_data(self.ntm.read1_hist, len(self.ntm.read1_hist))
        self.read2_hist_widget.update_data(self.ntm.read2_hist, len(self.ntm.read2_hist))
        self.weight_dyn_widget.update_data(self.ntm.read1_hist, 0) # Plot row 0
        
        in_val = [self.ntm.current_input] if is_writing else [0]
        out_val = [self.ntm.current_output] if not is_writing else [0]
        
        self.input_widget.update_data(in_val, 0 if is_writing else -1, is_active=is_writing)
        self.output_widget.update_data(out_val, 0 if not is_writing else -1, is_active=not is_writing)
        
        self.hidden_widget.update_data(self.ntm.hidden_state, is_active=True)
        self.read_vec1_widget.update_data(self.ntm.read_vector1, is_active=not is_writing)
        self.read_vec2_widget.update_data(self.ntm.read_vector2, is_active=not is_writing)
        self.key_widget.update_data(self.ntm.key_vector, is_active=is_writing)
        
        self.erase_widget.update_data(self.ntm.erase_vector, is_active=is_writing)
        self.add_widget.update_data(self.ntm.add_vector, is_active=is_writing)
        
        self.gate_w_widget.update_data(self.ntm.gate_w, is_active=is_writing)
        self.gate_r1_widget.update_data(self.ntm.gate_r1, is_active=not is_writing)
        self.gamma_w_widget.update_data(self.ntm.gamma_w, is_active=is_writing)
        
        self.shift_w_widget.update_data(self.ntm.shift_kernel_w, is_active=is_writing)
        self.shift_r_widget.update_data(self.ntm.shift_kernel_r, is_active=not is_writing)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1600, 900)
    window.show()
    sys.exit(app.exec())