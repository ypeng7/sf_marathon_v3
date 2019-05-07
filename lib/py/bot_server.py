# -*- coding: utf-8 -*-
# @author: Yue Peng
# @email: yuepaang@gmail.com
# @create by: 2018-12-13 11:08:45
import asyncio
import sys
from sanic import Sanic
from sanic import response
import time
# from bot import Bot
# from dfs import DFS
from ult_bot import Killer
# from greedy import Killer


app = Sanic(__name__)


port_ = int(sys.argv[1])
name = sys.argv[2]
# bot = Bot(name)
# bot = DFS(name)
bot = Killer(name)


@app.route('/start',methods=["POST"])
async def on_start(request):
	json = request.json
	print(json)
	return response.json({})

@app.route('/step',methods=["POST"])
async def on_step(request):
	json = request.json
	await asyncio.sleep(0.1)
	# state = json.loads(json_)
	bot.update_state(json)
	return response.json({'action': bot.bot_action()})

@app.route('/end',methods=["POST"])
async def on_end(request):
	json = request.json
	print(json)
	return response.json({})

def main():
	# app.run(host='10.118.105.243', port=port_)
	app.run(host='192.168.1.189', port=port_)

if __name__ == '__main__':
	main()
