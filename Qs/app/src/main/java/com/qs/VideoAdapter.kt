package com.qs

import android.content.Context
import android.net.Uri
import android.util.Log
import android.util.SparseArray
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.recyclerview.widget.RecyclerView

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore,
    private val onLongPress: () -> Unit
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    private val players = SparseArray<ExoPlayer>()
    private var currentPos = 0

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val playerView: PlayerView = view.findViewById(R.id.playerView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_video, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val uri = videos[position]
        var player = players.get(position)

        if (player == null) {
            // 【加速1】开启解码回退，防止 4K 报错
            val renderersFactory = DefaultRenderersFactory(context)
                .setEnableDecoderFallback(true)
                .setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER)

            // 【加速2】极其激进的缓冲策略：起播只要 500ms，后面疯狂缓存 50秒
            val loadControl = DefaultLoadControl.Builder()
                .setBufferDurationsMs(
                    15000, // 最小缓冲 15秒
                    50000, // 最大缓冲 50秒
                    500,   // 起播只要 500ms！立马出画面
                    2000   // 卡顿后重新缓冲 2秒
                )
                .build()

            player = ExoPlayer.Builder(context, renderersFactory)
                .setLoadControl(loadControl)
                .build()

            player.setMediaItem(MediaItem.fromUri(uri))
            player.repeatMode = Player.REPEAT_MODE_ONE
            
            // 【加速3】预加载但不自动播放
            player.prepare()

            val saved = historyStore.getPosition(uri.toString())
            if (saved > 0) player.seekTo(saved)

            player.addListener(object : Player.Listener {
                override fun onPlayerError(error: PlaybackException) {
                    super.onPlayerError(error)
                    Log.e("QsPlayer", "播放失败: ${error.errorCodeName}", error)
                }
            })
            players.put(position, player)
        }

        holder.playerView.player = player
        // 【加速4】提前把第一帧画出来，滑过去不黑屏
        holder.playerView.setKeepContentOnPlayerReset(true) 
        
        if (position == currentPos) {
            player.play()
        } else {
            player.pause()
        }

        holder.itemView.setOnClickListener {
            if (player.isPlaying) player.pause() else player.play()
        }
        
        holder.itemView.setOnLongClickListener {
            onLongPress()
            true
        }
    }

    override fun onViewRecycled(holder: VH) {
        super.onViewRecycled(holder)
        val pos = holder.bindingAdapterPosition
        if (pos != RecyclerView.NO_POSITION) {
            players.get(pos)?.let {
                historyStore.savePosition(videos[pos].toString(), it.currentPosition)
                it.release()
                players.remove(pos)
            }
        }
        holder.playerView.player = null
    }

    override fun getItemCount() = videos.size

    fun setCurrent(position: Int) {
        players.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
            it.pause()
        }
        currentPos = position
        players.get(position)?.play()
    }

    fun saveCurrentPosition() {
        players.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
        }
    }

    fun releaseAll() {
        for (i in 0 until players.size()) {
            val p = players.valueAt(i)
            historyStore.savePosition(videos[players.keyAt(i)].toString(), p.currentPosition)
            p.release()
        }
        players.clear()
    }
}
