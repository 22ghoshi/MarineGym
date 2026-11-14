"""
teleop_window.py - small keyboard-capture window for teleop.py.

teleop.py launches this as a separate process, so key presses go to THIS window
and never reach the Isaac Sim GUI (no shortcut conflicts). Protocol over pipes:

    stdout (to teleop.py):   READY | P <KEY> | R <KEY> | QUIT
    stdin  (from teleop.py): S <status text, '|' = newline> | QUIT

Not meant to be run on its own (but it will, and just print key events).
"""

import sys
import threading
import tkinter as tk

NAMES = {
    "space": "SPACE", "equal": "EQUAL", "plus": "EQUAL", "minus": "MINUS",
    "underscore": "MINUS", "KP_Add": "NUMPAD_ADD", "KP_Subtract": "NUMPAD_SUBTRACT",
}

HELP = (
    "W / S   surge fwd / back      Q / E   yaw left / right\n"
    "A / D   sway left / right     I / K   pitch nose down / up\n"
    "R / V   heave up / down       J / L   roll left / right\n"
    "= / -   gain up / down        T       toggle auto-level\n"
    "SPACE   hold: thrusters off   close window: quit teleop"
)


def norm(keysym: str) -> str:
    return keysym.upper() if len(keysym) == 1 else NAMES.get(keysym, keysym.upper())


root = tk.Tk()
root.title("ROV teleop - keep this window focused")
root.geometry("780x760")
root.minsize(560, 400)
root.configure(padx=14, pady=12)

FOCUSED_BG, UNFOCUSED_BG = "#123b3f", "#3a3a3a"
mono = ("DejaVu Sans Mono", 11)
mono_small = ("DejaVu Sans Mono", 10)

banner = tk.Label(root, text="FOCUSED - keys go to the ROV", font=("DejaVu Sans", 12, "bold"),
                  fg="white", bg=FOCUSED_BG, pady=6)
banner.pack(fill="x")
tk.Label(root, text=HELP, font=mono, justify="left", anchor="w").pack(fill="x", pady=(10, 6))
lbl_keys = tk.Label(root, text="held: -", font=mono, anchor="w", fg="#0a7")
lbl_keys.pack(fill="x")
lbl_status = tk.Label(root, text="waiting for simulator...", font=mono_small,
                      justify="left", anchor="nw")
lbl_status.pack(fill="both", expand=True, pady=(8, 0))

held, pending = set(), {}
state = {"status": "waiting for simulator...", "quit": False}


def send(msg: str):
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, ValueError):
        state["quit"] = True


def commit_release(k):
    pending.pop(k, None)
    if k in held:
        held.discard(k)
        send(f"R {k}")


def on_press(event):
    k = norm(event.keysym)
    if k in pending:                      # X11 auto-repeat: release+press pair -> ignore
        root.after_cancel(pending.pop(k))
        return
    if k not in held:
        held.add(k)
        send(f"P {k}")


def on_release(event):
    k = norm(event.keysym)
    if k in pending:
        root.after_cancel(pending.pop(k))
    pending[k] = root.after(40, lambda: commit_release(k))   # debounce auto-repeat


def release_all(_event=None):
    for k in list(pending):
        root.after_cancel(pending.pop(k))
    for k in list(held):
        held.discard(k)
        send(f"R {k}")


def on_focus_in(_event=None):
    banner.config(text="FOCUSED - keys go to the ROV", bg=FOCUSED_BG)


def on_focus_out(_event=None):
    release_all()                          # never leave a thruster stuck on
    banner.config(text="NOT FOCUSED - click here to drive", bg=UNFOCUSED_BG)


def on_close():
    send("QUIT")
    root.destroy()


def stdin_reader():
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.startswith("S "):
            state["status"] = line[2:].replace("|", "\n")
        elif line.strip() == "QUIT":
            break
    state["quit"] = True


def poll():
    lbl_keys.config(text="held: " + (" ".join(sorted(held)) or "-"))
    lbl_status.config(text=state["status"])
    if state["quit"]:
        root.destroy()
        return
    root.after(100, poll)


root.bind("<KeyPress>", on_press)
root.bind("<KeyRelease>", on_release)
root.bind("<FocusIn>", on_focus_in)
root.bind("<FocusOut>", on_focus_out)
root.protocol("WM_DELETE_WINDOW", on_close)

threading.Thread(target=stdin_reader, daemon=True).start()
root.after(100, poll)
root.after(200, root.focus_force)
send("READY")
root.mainloop()