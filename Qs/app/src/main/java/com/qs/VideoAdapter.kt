package com.qs

import android.content.Context
import android.net.Uri
import android.util.SparseArray
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.recyclerview.widget.RecyclerView

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore,
    private val onLongPress: () -> Unit
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    // 用一个简单的池子，防止频繁创建和销毁
    private val playerPool = SparseArray<ExoPlayer>()
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
        var player = playerPool.get(position)

        if (player == null) {
            player = ExoPlayer.Builder(context).build()
            player.setMediaItem(MediaItem.fromUri(uri))
            player.repeatMode = Player.REPEAT_MODE_ONE
            player.prepare()
            val saved = historyStore.getPosition(uri.toString())
            if (saved > 0) player.seekTo(saved)
            playerPool.put(position, player)
        }

        holder.playerView.player = player
        
        // 【核心防黑屏】保留最后一帧，绝不黑屏
        holder.playerView.setKeepContentOnPlayerReset(true)

        // 【关键逻辑】绑定的时候绝对不播，全部静音暂停！
        player.volume = 0f
        player.pause()

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
        // 只解绑，不销毁播放器（让池子管）
        holder.playerView.player = null
    }

    override fun getItemCount() = videos.size

    // 由 Activity 在滑动停止时调用
    fun onScrollIdle(position: Int) {
        currentPos = position
        // 将其他所有的播放器全部暂停
        for (i in 0 until playerPool.size()) {
            val p = playerPool.valueAt(i)
            p.pause()
            p.volume = 0f
        }
        // 只播当前这一个
        playerPool.get(position)?.let {
            it.volume = 1f
            it.play()
        }
        
        // 清理距离超过3的播放器，腾出内存
        val toRemove = mutableListOf<Int>()
        for (i in 0 until playerPool.size()) {
            val key = playerPool.keyAt(i)
            if (Math.abs(key - currentPos) > 3) {
                val p = playerPool.valueAt(i)
                historyStore.savePosition(videos[key].toString(), p.currentPosition)
                p.release()
                toRemove.add(key)
            }
        }
        toRemove.forEach { playerPool.remove(it) }
    }

    fun saveCurrentPosition() {
        playerPool.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
        }
    }

    fun releaseAll() {
        for (i in 0 until playerPool.size()) {
            val p = playerPool.valueAt(i)
            historyStore.savePosition(videos[playerPool.keyAt(i)].toString(), p.currentPosition)
            p.release()
        }
        playerPool.clear()
    }
}
