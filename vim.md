let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/.ycm_extra_conf.py'      "配置全局路径

let g:ycm_confirm_extra_conf=0   "每次直接加载该文件，不提示是否要加载

""""""""""""""""""""" Vundle
set nocompatible
filetype off


set rtp+=~/.vim/bundle/vundle
set rtp+={repository_root}/powerline/bindings/vim
call vundle#rc()


Bundle 'gmarik/vundle'


Bundle 'Valloric/YouCompleteMe'
Bundle 'Valloric/ListToggle'
Bundle 'scrooloose/syntastic'
Bundle 'scrooloose/nerdtree'
Bundle 'klen/python-mode'
Bundle 'croaky/vim-colors-github'
Bundle 'danro/rename.vim'
Bundle 'majutsushi/tagbar'
Bundle 'kchmck/vim-coffee-script'
Bundle 'kien/ctrlp.vim'
Bundle 'pbrisbin/vim-mkdir'
Bundle 'slim-template/vim-slim'
Bundle 'thoughtbot/vim-rspec'
Bundle 'tpope/vim-bundler'
Bundle 'tpope/vim-endwise'
Bundle 'tpope/vim-fugitive'
Bundle 'tpope/vim-rails'
Bundle 'tpope/vim-surround'
Bundle 'vim-ruby/vim-ruby'
Bundle 'vim-scripts/ctags.vim'
Bundle 'vim-scripts/matchit.zip'
Bundle 'vim-scripts/tComment'
Bundle "mattn/emmet-vim"
Bundle "Lokaltog/vim-powerline"
Bundle "godlygeek/tabular"
Bundle "msanders/snipmate.vim"
Bundle "jelera/vim-javascript-syntax"
Bundle "altercation/vim-colors-solarized"
Bundle "othree/html5.vim"
Bundle "xsbeats/vim-blade"
Bundle "Raimondi/delimitMate"
Bundle "groenewege/vim-less"
Bundle "evanmiller/nginx-vim-syntax"
Bundle "Lokaltog/vim-easymotion"
Bundle "tomasr/molokai"
Bundle 'davidhalter/jedi-vim'
Bundle 'plasticboy/vim-markdown'
Bundle 'suan/vim-instant-markdown'


filetype plugin indent on
""""""""""""""""""""" Vundle

"""""""""""""""""""""""""""""""""""""""基本设置""""""""""""""""""""""""""""""""""""""""
" 设定默认解码 
set fenc=utf-8 
set fencs=utf-8,usc-bom,euc-jp,gb18030,gbk,gb2312,cp936 

" 支持256色，使得vim配色支持终端
set t_Co=256

" C缩进
set smartindent 
set cindent

" 设置背景和字体
colorscheme molokai
set guifont=Source\ Code\ Pro\ 12 

" 不要使用vi的键盘模式，而是vim自己的 
set nocompatible 

" history文件中需要记录的行数 
set history=1000 

" 在处理未保存或只读文件的时候，弹出确认 
set confirm

" 与windows共享剪贴板 
set clipboard+=unnamed 

" 侦测文件类型 
filetype off 

" 为特定文件类型载入相关缩进文件 
filetype indent on 

" 带有如下符号的单词不要被换行分割 
set iskeyword+=_,$,@,%,#,- 

" 语法高亮
syntax enable
syntax on 

"隐藏GUI的工具栏
set guioptions=P

" 不要备份文件 
set nobackup 

" 不要生成swap文件
setlocal noswapfile 

" 字符间插入的像素行数目 
set linespace=0 

" 在状态行上显示光标所在位置的行号和列号 
set ruler 

" 命令行（在状态行下）的高度，默认为1，这里是2 
set cmdheight=2 

" 使回格键（backspace）正常处理indent, eol, start等 
set backspace=indent,eol,start

" 允许backspace和光标键跨越行边界 
set whichwrap+=<,>,b,s,[,]


" 高亮显示匹配的括号 
set showmatch 

" 匹配括号高亮的时间（单位是十分之一秒） 
set matchtime=5 

" 在搜索的时候忽略大小写 
set ignorecase 

" 不要高亮被搜索的句子（phrases） 
set nohlsearch 

" 在搜索时，输入的词句的逐字符高亮（类似firefox的搜索） 
set incsearch 

" 光标移动到buffer的顶部和底部时保持3行距离,窗口滚动最小距离 
set scrolloff=3 

" 2为总显示最后一个窗口的状态行
" 设为1则窗口数多于一个的时候显示最后一个窗口的状态行；
" 0不显示最后一个窗口的状态行 
set laststatus=2 

" 继承前一行的缩进方式，特别适用于多行注释 
""set autoindent 

" 显示行号
set number

" 制表符为4 
set tabstop=4 

" 统一缩进为4 
set softtabstop=4 
set shiftwidth=4 

" 不要用空格代替制表符 
set noexpandtab 

" 不要换行 
" set nowrap 
" set sidescroll=10

" 在行和段开始处使用制表符 
set smarttab 

" Ctrl+A全选，Ctrl+C复制，Ctrl+V粘贴
map <C-A> ggvG$ 
imap <C-A> <Esc>ggvG$
vmap <C-C> "+y<Esc>
map <C-V> "+p
imap <C-V> <Esc>"+pa

" 括号等的自动匹配
inoremap ( ()<Esc>i
inoremap [ []<Esc>i
inoremap { {}<Esc>i
inoremap ' ''<Esc>i
inoremap " ""<Esc>i

" 设置<leader>和<localleader>
let mapleader = ","
let maplocalleader = "."

" 可以折叠 
set foldenable 
set foldmethod=manual 

" 自动更新.vimrc
map <leader>vo <Esc>:vsp ~/.vimrc<CR>

"""""""""""""""""""""""""""""""""""""""基本设置""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""
"   instant markdown slow插件               "
"""""""""""""""""""""""""""""""""""""
let g:instant_markdown_slow = 1
let g:instant_markdown_autostart = 0
let g:instant_markdown_open_to_the_world = 1
let g:instant_markdown_allow_unsafe_content = 1
let g:instant_markdown_allow_external_content = 0
"""""""""""""""""""""""""""""""""""""
"   jedi插件               "
"""""""""""""""""""""""""""""""""""""
let g:jedi#auto_initialization = 0

let g:jedi#auto_vim_configuration = 0

let g:jedi#use_tabs_not_buffers = 1

let g:jedi#use_splits_not_buffers = "left"

let g:jedi#popup_on_dot = 0

let g:jedi#popup_select_first = 0

let g:jedi#show_call_signatures = "1"
let g:jedi#goto_command = "<leader>d"
let g:jedi#goto_assignments_command = "<leader>g"
let g:jedi#goto_definitions_command = ""
let g:jedi#documentation_command = "K"
let g:jedi#usages_command = "<leader>n"
let g:jedi#completions_command = "<C-Space>"
let g:jedi#rename_command = "<leader>r"
let g:jedi#completions_enabled = 0

"""""""""""""""""""""""""""""""""""""
"   NERD Tree插件               "
"""""""""""""""""""""""""""""""""""""
nmap <F2> :NERDTreeToggle<cr>
" F2打开目录树
let g:NERDTreeWinSize=30
" 目录树横向大小
let NERDChristmasTree=0

let NERDTreeChDirMode=2
let NERDTreeIgnore=['\~$', '\.pyc$', '\.swp$']
let NERDTreeShowBookmarks=1
let NERDTreeWinPos="left"
" Automatically open a NERDTree if no files where specified
autocmd vimenter * if !argc() | NERDTree | endif
" Close vim if the only window left open is a NERDTree
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTreeType") && b:NERDTreeType == "primary") | q | endif


"""""""""""""""""""""""""""""""""""""
"   Tagbar   插件               "
"""""""""""""""""""""""""""""""""""""
map tb <Esc>:Tagbar <CR>
" tb用于打开tagbar
let g:tagbar_width=30
let g:tagbar_autofocus=1
let g:Powerline_symbols = 'fancy'



"""""""""""""""""""""""""""""""""""""
"            主题颜色               "
"""""""""""""""""""""""""""""""""""""
let g:molokai_original = 1
let g:rehash256 = 1
" Color scheme
colorscheme molokai
highlight NonText guibg=#060606
highlight Folded  guibg=#0A0A0A guifg=#9090D0
set rtp+={repository_root}/powerline/bindings/vim



"""""""""""""""""""""""""""""""""""""
"           ctags插件               "
"""""""""""""""""""""""""""""""""""""
set tags=./tags,./../tags,./*/tags 
"--这样的设置可以生成当前目录，上级目录以及当前目录的所有子目录的tags文件
map <F7> <Esc>:!sudo ctags -R <CR><CR>


"""""""""""""""""""""""""""""""""""""
"           powerline插件           "
"""""""""""""""""""""""""""""""""""""
set guifont=PowerlineSymbols\ for\ Powerline
set nocompatible
set laststatus=2
set t_Co=256
let g:Powerline_symbols = 'fancy'
let Powerline_symbols='compatible'



"""""""""""""""""""""""""""""""""""""
"           YouComPleteMe插件       "
"""""""""""""""""""""""""""""""""""""

let mapleader = '\'
nnoremap <leader>gg :YcmCompleter GoToDeclaration<CR> 
" 跳转到声明处
nnoremap <leader>gd :YcmCompleter GoToDefinition<CR> 



""""""""""""""""""""""""""""""""""""""""C语言的编译运行"""""""""""""""""""""""""""""""""""""""""
" <F5>编译C/py语言，<F6>运行
augroup ccompile
    autocmd Filetype c map <F5> <Esc>:w<CR>:!gcc % -std=c99 -g -o %< -lm <CR>
    autocmd Filetype cpp map <F5> <Esc>:w<CR>:!g++ % -std=c++11 -g -o %< -lm <CR>
    autocmd Filetype python map <F5> <Esc>:w<CR>:!python2 % <CR>
    autocmd Filetype python map <F5> <Esc>:w<CR>:!python3 % <CR>
augroup END

augroup crun
    autocmd Filetype c map <F6> <Esc>:! ./%< <CR>
    autocmd Filetype cpp map <F6> <Esc>:! ./%< <CR>
augroup END

" 整行注释
augroup comment
    autocmd Filetype c noremap <buffer> <localleader>/ I//<Esc>
    autocmd Filetype cpp noremap <buffer> <localleader>/ I//<Esc>
    autocmd Filetype h noremap <buffer> <localleader>/ I//<Esc>
augroup END

augroup nocomment
    autocmd Filetype c noremap <buffer> <localleader>. ^xx
    autocmd Filetype cpp noremap <buffer> <localleader>. ^xx
    autocmd Filetype h noremap <buffer> <localleader>. ^xx
augroup END


" 大括号补全
autocmd Filetype c,cpp,h inoremap {<CR> {<CR>}<Esc>O

""""""""""""""""""""""""""""""""""""""""C语言的编译运行"""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""新文件标题""""""""""""""""""""""""""""""""""""""""""""""""
"新建.c,.h,.sh,.java文件，自动插入文件头 
autocmd BufNewFile *.py,*.cpp,*.[ch],*.sh,*.java exec ":call SetTitle()" 
""定义函数SetTitle，自动插入文件头 
func SetTitle() 
    "如果文件类型为.sh文件 
    if &filetype == 'sh' 
        call setline(1,"\#########################################################################") 
        call append(line("."), "\# File Name: ".expand("%")) 
        call append(line(".")+1, "\# Author: lbz") 
        call append(line(".")+2, "\# mail: 1647615313@qq.com") 
        call append(line(".")+3, "\# Created Time: ".strftime("%c")) 
        call append(line(".")+4, "\#########################################################################") 
        call append(line(".")+5, "\#!/bin/bash") 
        call append(line(".")+6, "") 
    elseif &filetype == 'python'
        call setline(1,"\#########################################################################") 
        call append(line("."), "\# File Name: ".expand("%")) 
        call append(line(".")+1, "\# Author: lbz") 
        call append(line(".")+2, "\# mail: 120498304@qq.com") 
        call append(line(".")+3, "\# Created Time: ".strftime("%c")) 
        call append(line(".")+4, "\#########################################################################") 
        call append(line(".")+5, "\#!/usr/bin/env python3") 
        call append(line(".")+6, "\#! -*- coding: utf-8 -*-") 
    else 
        call setline(1, "/*************************************************************************") 
        call append(line("."), "    > File Name: ".expand("%")) 
        call append(line(".")+1, "    > Author: King") 
        call append(line(".")+2, "    > Mail: arturiapendragon_1@163.com ") 
        call append(line(".")+3, "    > Created Time: ".strftime("%c")) 
        call append(line(".")+4, " ************************************************************************/") 
        call append(line(".")+5, "")
    endif
    if &filetype == 'cpp'
        call append(line(".")+6, "#include<iostream>")
        call append(line(".")+7, "using namespace std;")
        call append(line(".")+8, "")
    endif
    if &filetype == 'c'
        call append(line(".")+6, "#include<stdio.h>")
        call append(line(".")+7, "")
    endif
    "新建文件后，自动定位到文件末尾
    autocmd BufNewFile * normal G
endfunc 

""""""""""""""""""""""""""""""""""""""新文件标题""""""""""""""""""""""""""""""""""""""""""""""""
