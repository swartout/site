---
title: Terminal Stuff
---

# Terminal Stuff

---

## TMUX

**TMUX Cheat Sheet**

I setup TMUX to work with nvim using [vim-tmux-navigator](https://github.com/christoomey/vim-tmux-navigator), so my defaults for moving left, right, up, and down are vim defaults.

- Start tmux using `tmux`
- Think of windows as "screens" and panes as tabs
- Start tmux with `name` using `tmux new -s name`
- List tmux sessions: `tmux ls`
- Attach to a session: `tmux a #`
- Attach to a named session: `tmux a -t name`
- Kill a session: `tmux kill-session -t name`
- Kill tmux server: `tmux kill-server`
- Use `Ctrl+hjkl` to switch between panes
- Press `Ctrl-a` to prefix tmux keyboard shortcuts (remapped)
- `%` to split screen horizontally
- `"` to split screen vertically
- `d` to detach (leaving it running in background)
- `x` to kill current pane
- `?` to list all keybindings
- `{` move current pane left
- `}` move current pane right
- `o` go to next pane
- `z` toggle full-screen mode for current pane

References: [TMUX Guide](https://tmuxguide.readthedocs.io/en/latest/tmux/tmux.html)

---

## Vim

- Use `i` to do an action within a motion
    - e.g. `ciw` to change the word
- Use `c` to change within motion
- Use `p` to denote paragraph
- Use `:set` to access or modify values
    - Append option with `?` to see the current value
- Buffers: an open file in memory
    - Three states:
        - Active: buffer is displayed in a window
        - Hidden: buffer exists, but is not displayed
        - Inactive: buffer is not displayed and does not link to a file
    - `:buffers` or `:ls` to list all the current buffers
        - Displays buffer ID, state, name, and cursor line
        - Hidden buffers won't be displayed, either `set hidden` or append `!` to command
    - Most buffer commands start with `:b`
        - `:bn` to move to next buffer
        - `:bp` to move to previous buffer
        - `:bf` to move to first buffer
        - `:bl` to move to last buffer
        - `Ctrl-^` to switch to alternate buffer
        - Run command on all buffers: `:bufdo <command>`
        - Use `:b <name>` to switch to buffer with name using partial name matching
    - Create and delete buffers:
        - `:badd <filename>` adds `<filename>` to the buffer list
        - `:bdelete <ID_or_name>` deletes *all* buffers in `<ID_or_name>`
        - `:%bdelete` deletes all buffers
- Windows: where vim can display the content of a buffer
    - To create a new window use `:new`
        - `CTRL-W s` to split the current window horizontally
        - `CTRL-W v` to split the current window vertically
        - `CTRL-W n` to split the current window horizontally *and* edit a new file
        - `CTRL-W ^` to split the current window with the alternate file (buffer with `#` indicator)
        - `<buffer_ID>CTRL-W ^` split current window with `<buffer_ID>`
    - To switch between windows:
        - `CTRL-W CTRL-W` to switch to last (alternate?) window
        - `CTRL-<hjkl>` to move left, down, up, or right (remap)
    - To rotate windows:
        - `CTRL-W r` rotate the windows
        - `CTRL-W x` exchange with next window
    - To resize windows:
        - `CTRL-W =` resize windows to fit on screen with the same size
        - `CTRL-W +` increase window height
        - `CTRL-W -` decrease window height
        - `CTRL-W <` decrease window width
        - `CTRL-W >` increase window width
    - Exit windows with `:q` or `:q!` (these commands don't quit vim, but quit a window!)
- To edit a new file use `:e`

---

## Lua + Neovim

- [Lua basics:](https://learnxinyminutes.com/docs/lua/) honestly pretty similar to JS/Python
- comments are `--`, multi-line is two brackets `[[comment here]]`
- use `if condition then` for if statements
    - finish if statements with `end`
- blocks use `do/end`
- not equals is: `~=`
- finish functions with `end` after `return`
- most everything else is either unimportant for what I care about right now, or something I'll google

- [Neovim lua config:](https://vonheikemen.github.io/devlog/tools/build-your-first-lua-config-for-neovim/)
- To edit global settings, use the global variable/module `vim`
    - i.e. `vim.opt.option_name = value`
- `lua/plugins.lua` contains the config file for loading plugins
